"""Full-gallery CoVR-2 subset evaluation; no baseline evaluation is imported."""
import csv
from collections import OrderedDict
import json
from pathlib import Path
from zipfile import BadZipFile
import numpy as np
from tqdm import tqdm

from visual_retrieval.common import file_signature,write_json
from visual_retrieval.composed.gallery import FORMAT,token_path,validate_tokens
from visual_retrieval.composed.scoring import score_video
from visual_retrieval.composed.webvid_covr import QUERY_FORMAT,read_query
from .ranking_metrics import metrics_from_ranks
from visual_retrieval.features.pipeline import prefetch_sources


class TokenCache:
    """Byte-bounded LRU of validated arrays; accessed only on the main thread."""
    def __init__(self,max_bytes):
        self.max_bytes=max_bytes
        self.bytes=0
        self.arrays=OrderedDict()

    def get(self,id):
        value=self.arrays.get(id)
        if value is not None:self.arrays.move_to_end(id)
        return value

    def put(self,id,value):
        if value.nbytes>self.max_bytes:return
        old=self.arrays.pop(id,None)
        if old is not None:self.bytes-=old.nbytes
        while self.arrays and self.bytes+value.nbytes>self.max_bytes:
            _,removed=self.arrays.popitem(last=False)
            self.bytes-=removed.nbytes
        self.arrays[id]=value
        self.bytes+=value.nbytes


def clear_reports(output):
    output=Path(output)
    for name in ('evaluation.json','query_ranks.csv','predictions.jsonl'):
        (output/name).unlink(missing_ok=True)


def evaluate(data,gallery_dir,query_dir,output,gallery='targets',top_k=10,allow_incomplete=False,
             validation_workers=4,gallery_cache_mb=512):
    if type(top_k) is not int or top_k<1:
        raise ValueError('Use a positive exported top-k; target ranks always use the full gallery.')
    if type(validation_workers) is not int or validation_workers<0 or type(gallery_cache_mb) is not int or gallery_cache_mb<0:
        raise ValueError('Validation workers and gallery cache MB must be nonnegative integers.')
    gallery_dir,query_dir,output=map(Path,(gallery_dir,query_dir,output))
    output.mkdir(parents=True,exist_ok=True)
    clear_reports(output)
    print(f'Starting CoVR-2 evaluation on CPU: {len(data.queries)} queries. '
          'Gallery validation and full-gallery scoring follow; no GPU model is loaded.',flush=True)
    saved=json.loads((gallery_dir/'manifest.json').read_text(encoding='utf-8'))
    identity=saved['identity']
    query_config=json.loads((query_dir/'run_config.json').read_text(encoding='utf-8'))
    query_manifest=json.loads((query_dir/'manifest.json').read_text(encoding='utf-8'))
    if (identity.get('format')!=FORMAT or identity.get('frames')!=15
            or query_config.get('query_format')!=QUERY_FORMAT
            or identity.get('base_sha256')!=query_config.get('base_sha256')
            or identity.get('precision')!=query_config.get('precision')
            or any(query_manifest.get(k)!=v for k,v in query_config.items())):
        raise ValueError('CoVR-2 gallery/query checkpoint or precision provenance differs; re-extract matching features.')
    rows=saved['gallery_items']
    if len({row['item_id'] for row in rows})!=len(rows):raise ValueError('Duplicate gallery cache IDs.')
    metadata={row['item_id']:row for row in rows}
    items={item['item_id']:item for item in data.items}
    selected=data.candidate_ids(gallery)
    available,excluded_gallery=[],[]
    gallery_errors={}
    token_cache=TokenCache(gallery_cache_mb*1024*1024)
    def validate_candidate(id):
        try:
            row=metadata.get(id)
            if row is None:raise ValueError('Gallery item was not extracted.')
            stat=file_signature(items[id]['path'])
            if row.get('source_signature')!={k:stat[k] for k in ('size','mtime_ns')}:
                raise ValueError('Gallery video changed; rerun extraction.')
            if Path(row['path']).resolve()!=items[id]['path']:
                raise ValueError('Gallery source path differs from the selected subset.')
            tokens=validate_tokens(np.load(token_path(gallery_dir,id),allow_pickle=False))
            return id,tokens,None
        except (ValueError,OSError,KeyError,EOFError,BadZipFile) as error:
            return id,None,str(error)
    stream=prefetch_sources(selected,validate_candidate,workers=validation_workers)
    print(f'Validation workers: {validation_workers}; gallery token cache: {gallery_cache_mb} MB.',flush=True)
    try:
        for id,tokens,error in tqdm(stream,total=len(selected),desc='Validating CoVR-2 gallery',unit='video'):
            if error is not None:
                excluded_gallery.append(id);gallery_errors[id]=error
            else:
                available.append(id)
                token_cache.put(id,tokens)
    finally:
        stream.close()
    def gallery_tokens(id):
        tokens=token_cache.get(id)
        if tokens is None:
            tokens=validate_tokens(np.load(token_path(gallery_dir,id),allow_pickle=False))
            token_cache.put(id,tokens)
        return tokens
    if excluded_gallery and not allow_incomplete:
        raise ValueError(f'CoVR-2 gallery is incomplete/stale: {excluded_gallery[:10]}; re-extract or use --allow-incomplete.')
    if not available:raise ValueError('No valid CoVR-2 gallery candidates.')
    print(f'Scoring {len(data.queries)} queries against up to {len(available)} candidates each. '
          'The first query may take a while; the progress bar updates after each query.',flush=True)
    records,excluded_queries=[],[]
    for query in tqdm(data.queries,desc='Evaluating CoVR-2 queries',unit='query'):
        try:
            if query['query_id'] not in query_manifest['successful_query_ids']:
                raise ValueError('No completed query encoding.')
            if query['video_id'] not in available:
                raise ValueError('Ground-truth target is missing from the valid gallery.')
            query_tokens,text=read_query(query_dir,query)
            ids=[id for id in available if id!=query['reference_id']]
            scores=np.array([score_video(query_tokens,
                gallery_tokens(id),text) for id in ids],dtype=np.float32)
            if not np.isfinite(scores).all():raise ValueError('Nonfinite CoVR-2 score.')
            order=np.argsort(-scores,kind='stable')
            target=ids.index(query['video_id'])
            rank=int(np.flatnonzero(order==target)[0])+1
            predicted=ids[int(order[0])]
            candidates=[dict(rank=i+1,video_id=ids[int(index)],score=float(scores[index]),
                             is_ground_truth=ids[int(index)]==query['video_id'])
                        for i,index in enumerate(order[:top_k])]
            records.append(dict(query_id=query['query_id'],row_index=query['row_index'],reference_id=query['reference_id'],
                modification=query['text'],reference_image=str(query['reference_image']),
                ground_truth_video_id=query['video_id'],predicted_video_id=predicted,rank=rank,
                reciprocal_rank=1/rank,is_correct=predicted==query['video_id'],candidate_count=len(ids),
                ground_truth_score=float(scores[target]),predicted_score=float(scores[order[0]]),top_k=candidates))
        except (ValueError,OSError,KeyError,EOFError,BadZipFile) as error:
            if not allow_incomplete:
                raise ValueError(f'Cannot evaluate {query["query_id"]}: {error}') from error
            excluded_queries.append(dict(query_id=query['query_id'],error=str(error)))
    if not records:raise ValueError('No evaluable CoVR-2 queries; no metrics written.')
    ranks=np.array([r['rank'] for r in records],np.int64)
    metrics=metrics_from_ranks(ranks)
    metrics.update(r50=float(100*np.mean(ranks<=50)),median_rank=float(np.median(ranks)),mean_rank=float(np.mean(ranks)))
    report=dict(model='covr2',evaluated_models=['covr2'],dataset='WebVid-CoVR',split=data.split,
        subset_evaluation=True,subset_sha256=data.subset_sha256,gallery_mode=gallery,
        gallery_size=len(available),gallery_size_requested=len(selected),gallery_ids=available,
        num_queries=len(records),num_queries_requested=len(data.queries),download_requested_samples=data.requested_samples,
        num_downloaded_samples=data.downloaded_samples,download_complete=data.download_complete,
        download_failed_rows=len(data.download_failures),download_failures=data.download_failures,
        excluded_gallery_ids=excluded_gallery,gallery_errors=gallery_errors,excluded_queries=excluded_queries,
        complete=data.download_complete and not excluded_gallery and not excluded_queries,
        evaluation_complete=not excluded_gallery and not excluded_queries,allow_incomplete=allow_incomplete,
        metrics=metrics,recall_unit='percent',candidate_count_min=min(r['candidate_count'] for r in records),
        candidate_count_max=max(r['candidate_count'] for r in records),reference_excluded=True,
        reference_mode='downloaded_midpoint_image',frame_weighting='runtime_modification_itc',
        frame_weight_temperature=.1,token_storage='float16_or_float32',score_accumulation='float32',
        tie_policy='stable_selected_gallery_order',query_encoder=query_config,
        protocol_note='Existing runtime CoVR-2 implementation on a downloaded subset; not official benchmark reproduction.')
    fields=['query_id','row_index','reference_id','ground_truth_video_id','modification','predicted_video_id',
            'rank','reciprocal_rank','is_correct','candidate_count','ground_truth_score','predicted_score']
    table=output/'query_ranks.csv.tmp';predictions=output/'predictions.jsonl.tmp'
    with table.open('w',encoding='utf-8',newline='') as handle, predictions.open('w',encoding='utf-8') as stream:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for record in records:
            writer.writerow(record)
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    table.replace(output/'query_ranks.csv');predictions.replace(output/'predictions.jsonl')
    write_json(output/'evaluation.json',report)
    print(f'CoVR-2 only: {len(records)} queries / {len(available)} gallery videos; '
          f'R@1 {metrics["r1"]:.2f}%, R@5 {metrics["r5"]:.2f}%, R@10 {metrics["r10"]:.2f}%, MRR {metrics["mrr"]:.4f}',flush=True)
    return report
