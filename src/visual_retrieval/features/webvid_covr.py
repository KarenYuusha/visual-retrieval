"""Encode WebVid-CoVR with the baseline adapters; deliberately no evaluation."""
import gc
import json
import time
from pathlib import Path
from zipfile import BadZipFile
import numpy as np

from visual_retrieval.common import file_signature,normalize_features,write_json
from visual_retrieval.data.video_sampling import read_video_frames
from visual_retrieval.data.webvid_covr import PROTOCOL,serializable_queries
from visual_retrieval.models.registry import MODEL_INFO,checkpoint_signature,resolve_model_files,create_encoder
from .cache import atomic_npz,cached_video,ensure_cache_config
from .pipeline import prefetch_sources


def release_gpu():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_baseline(data,model,output,checkpoint=None,device='cuda',precision='auto',
                     text_batch_size=32,allow_incomplete=False,factory=create_encoder,resolver=resolve_model_files,
                     decode_workers=2,video_batch_size=None):
    from visual_retrieval.models.base import resolve_precision
    if model not in MODEL_INFO or text_batch_size<1:
        raise ValueError('Use a baseline model and a positive text batch size.')
    if decode_workers<0 or (video_batch_size is not None and video_batch_size<1):
        raise ValueError('Decoder workers must be nonnegative and video batch size positive.')
    video_batch_size=video_batch_size or (1 if model=='internvideo2' else 4)
    effective=resolve_precision(device,precision)
    checkpoint,tokenizer=resolver(model,checkpoint,None)
    info=MODEL_INFO[model]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config=dict(model_name=model,model_id=info['model_id'],source_commit=info['source_commit'],
        preprocessing=info['preprocessing'],checkpoint=checkpoint_signature(checkpoint),tokenizer=tokenizer,
        device=device,precision=precision,effective_precision=effective,protocol=PROTOCOL)
    ensure_cache_config(output,config)
    cache=output/'video_cache';cache.mkdir(exist_ok=True)
    encoder=None
    def active():
        nonlocal encoder
        if encoder is None:
            encoder=factory(model,checkpoint,device,precision,tokenizer)
            if encoder.effective_precision!=effective:
                raise ValueError('Actual encoder precision differs from cache provenance.')
        return encoder
    features,failed={},[]
    reused=0
    started=time.perf_counter()
    pending=[]
    batches=0
    stream=None
    def decode(source):
        item,signature=source
        try:
            sample=read_video_frames(item['path'],count_frames=info['num_frames'])
        except (ValueError,OSError) as error:
            sample=error
        return item,signature,sample
    def flush():
        nonlocal batches
        if not pending:return
        # Inference and all model access stay on the main thread.
        encoded=normalize_features(active().encode_videos([sample[0] for _,_,sample in pending]))
        if encoded.shape!=(len(pending),512):raise ValueError('Expected baseline video embeddings [B,512].')
        for (item,signature,sample),feature in zip(pending,encoded):
            id=item['item_id']
            atomic_npz(cache/f'{id}.npz',feature=feature,source_signature=json.dumps(signature),
                       frame_indices=np.asarray(sample[1]),decoded_frames=sample[2])
            features[id]=feature
            print(f'{model}: encoded video {id}',flush=True)
        batches+=1
        pending.clear()
    try:
        sources=[]
        for item in data.items:
            id,path=item['item_id'],item['path']
            try:
                signature=file_signature(path)
                feature=cached_video(cache/f'{id}.npz',signature)
                if feature is not None:
                    features[id]=feature;reused+=1;continue
            except (ValueError,OSError) as error:
                failed.append(dict(video_id=id,error=str(error)));continue
            sources.append((item,signature))
        print(f'{model}: decode workers={decode_workers}, video batch={video_batch_size}, cached={reused}',flush=True)
        stream=prefetch_sources(sources,decode,workers=decode_workers)
        for item,signature,sample in stream:
            if isinstance(sample,Exception):
                failed.append(dict(video_id=item['item_id'],error=str(sample)));continue
            pending.append((item,signature,sample))
            if len(pending)>=video_batch_size:flush()
        flush()
        if not features:
            write_json(output/'failed_videos.json',failed)
            raise ValueError(f'No {model} videos encoded; see failed_videos.json.')
        # Labels remain query-aligned. No baseline similarity matrix or ranks are computed.
        query_key=json.dumps([(q['query_id'],q['text'],q['reference_id'],q['video_id']) for q in data.queries])
        text_path=output/'text_cache.npz'
        texts=None
        if text_path.is_file():
            try:
                with np.load(text_path,allow_pickle=False) as stored:
                    if str(stored['query_key'])==query_key:
                        candidate=normalize_features(stored['text_features'])
                        if candidate.shape==(len(data.queries),512):texts=candidate
            except (ValueError,OSError,KeyError,EOFError,BadZipFile):pass
        if texts is None:
            texts=np.empty((len(data.queries),512),np.float32)
            for start in range(0,len(texts),text_batch_size):
                batch=data.queries[start:start+text_batch_size]
                values=normalize_features(active().encode_text([q['text'] for q in batch]))
                if values.shape!=(len(batch),512):raise ValueError('Expected baseline text embeddings [B,512].')
                texts[start:start+len(batch)]=values
            atomic_npz(text_path,text_features=texts,query_key=query_key)
        ids=[item['item_id'] for item in data.items if item['item_id'] in features]
        keep=[i for i,q in enumerate(data.queries) if q['video_id'] in features]
        queries=[data.queries[i] for i in keep]
        atomic_npz(output/'features.npz',video_features=np.stack([features[id] for id in ids]),
            video_ids=np.array(ids),text_features=texts[keep],text_video_ids=np.array([q['video_id'] for q in queries]),
            query_ids=np.array([q['query_id'] for q in queries]),reference_ids=np.array([q['reference_id'] for q in queries]))
        manifest=dict(**config,dataset='webvid_covr',split=data.split,subset_sha256=data.subset_sha256,
            gallery_items=[dict(item,path=str(item['path'])) for item in data.items],
            requested_video_ids=[item['item_id'] for item in data.items],successful_video_ids=ids,
            failed_videos=failed,complete=not failed,num_queries=len(queries),num_queries_requested=len(data.queries),
            reused_video_embeddings=reused,purpose='encoding_only_no_baseline_evaluation',
            efficiency=dict(decode_workers=decode_workers,video_batch_size=video_batch_size,
                video_batches=batches,new_video_embeddings=len(features)-reused,
                elapsed_seconds=time.perf_counter()-started))
        elapsed=manifest['efficiency']['elapsed_seconds']
        print(f'{model}: {len(features)-reused} new videos, {reused} reused; {elapsed:.1f}s',flush=True)
        write_json(output/'manifest.json',manifest)
        write_json(output/'queries.json',serializable_queries(queries))
        write_json(output/'failed_videos.json',failed)
        if failed and not allow_incomplete:
            raise ValueError(f'{model}: incomplete extraction; retry or explicitly use --allow-incomplete.')
        return manifest
    finally:
        if stream is not None:stream.close()
        pending.clear()
        encoder=None
        release_gpu()
