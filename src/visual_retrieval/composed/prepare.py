"""Prepare separate CoVR-2 gallery tokens from an existing extraction manifest."""
import argparse
import gc
import json
from pathlib import Path
import os
import time
import numpy as np
from visual_retrieval.data.video_sampling import read_video_segments
from visual_retrieval.models.covr2_encoder import CoVR2Encoder
from visual_retrieval.features.pipeline import prefetch_sources
from .gallery import FORMAT, atomic_json, digest, token_path, validate_tokens


def prepare(root, source_manifest, checkpoint, output=None, device='cuda', limit=None, encoder_factory=CoVR2Encoder,
            decode_workers=2, frame_batch_size=1):
    if decode_workers<0 or frame_batch_size<1:
        raise ValueError('Decoder workers must be nonnegative and frame batch size positive.')
    started=time.perf_counter()
    root, checkpoint = Path(root), Path(checkpoint)
    output = Path(output or root / 'features' / 'covr2')
    source = json.loads(Path(source_manifest).read_text(encoding='utf-8'))
    rows = source.get('gallery_items')
    if not rows:
        # Older unsegmented galleries may record IDs only.
        if source.get('protocol') == 'combined_subset_activitynet_segments':
            raise ValueError('ActivityNet requires gallery_items with segment boundaries.')
        rows = [dict(item_id=i, source_video_id=i, path=i+'.mp4', start=None, end=None)
                for i in source['requested_video_ids']]
    if limit is not None:
        rows = rows[:limit]
    if len({r['item_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate gallery IDs.')
    identity = dict(format=FORMAT, base_sha256=digest(checkpoint), frames=15, precision='fp32' if device=='cpu' else 'fp16')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    old = {}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['identity'] != identity:
            raise ValueError('Existing CoVR-2 cache has different provenance. Use a new output folder.')
    (output / 'tokens').mkdir(exist_ok=True)
    previous = {r['item_id']: r for r in old.get('gallery_items',[])}
    rows = [dict(r) for r in rows]
    manifest = dict(identity=identity, gallery_items=rows, failures={}, sampling=old.get('sampling',{}))
    grouped = {}
    for row in rows:
        # Use the same canonical filename rules as the demo registry.
        from visual_retrieval.demo.service import _item_metadata
        from visual_retrieval.demo.config import DatasetSpec
        item = _item_metadata(DatasetSpec('prepare','prepare',root,{}),row['item_id'],row,
                              source.get('protocol') == 'combined_subset_activitynet_segments')
        target = token_path(output, row['item_id'])
        try:
            stat = Path(item['path']).stat()
            row['source_signature'] = dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
        except OSError:
            row['source_signature'] = None
        if target.exists() and previous.get(row['item_id']) != row:
            target.unlink()
        if target.exists():
            try:
                validate_tokens(np.load(target,allow_pickle=False));continue
            except (ValueError,OSError,EOFError):
                target.unlink()
        grouped.setdefault(item['path'], []).append((row,item))
    # Metadata follows invalidation, before inference; an interruption is resumable.
    atomic_json(manifest_path, manifest)
    def decode(group):
        path=group[0][1]['path']
        try:
            samples=read_video_segments(path, [(i['start'],i['end']) for _,i in group],15)
        except (ValueError,OSError) as error:
            samples=[error]*len(group)
        return group,samples
    encoder=None
    stream=prefetch_sources(grouped.values(),decode,workers=decode_workers)
    print(f'CoVR-2 gallery: decode workers={decode_workers}, frame batch={frame_batch_size}, '
          f'uncached videos={len(grouped)}',flush=True)
    try:
        if grouped:
            encoder=encoder_factory(checkpoint,device=device,frame_batch_size=frame_batch_size)
        for group,samples in stream:
            for (row, _), sample in zip(group, samples):
                try:
                    if isinstance(sample,Exception): raise sample
                    tokens = validate_tokens(encoder.encode_frames(sample[0])).astype(np.float16)
                    manifest['sampling'][row['item_id']] = dict(frame_indices=sample[1],decoded_frame_count=sample[2])
                    target = token_path(output,row['item_id'])
                    with target.with_suffix('.tmp').open('wb') as f: np.save(f,tokens,allow_pickle=False)
                    os.replace(target.with_suffix('.tmp'),target)
                    print('Saved',row['item_id'],flush=True)
                except (ValueError,OSError) as error:
                    manifest['failures'][row['item_id']] = str(error)
                    print('Skipped',row['item_id'],error,flush=True)
            atomic_json(manifest_path,manifest)
    finally:
        stream.close()
        del encoder
        gc.collect()
    manifest['efficiency']=dict(decode_workers=decode_workers,frame_batch_size=frame_batch_size,
        elapsed_seconds=time.perf_counter()-started)
    atomic_json(manifest_path,manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--limit',type=int)
    p.add_argument('--decode-workers',type=int,default=2,help='CPU decoding threads; 0 disables prefetch.')
    p.add_argument('--frame-batch-size',type=int,default=1,help='CoVR-2 frames per inference; increase if VRAM permits.')
    a=p.parse_args()
    if a.limit is not None and a.limit < 1: p.error('--limit must be positive')
    if a.decode_workers<0 or a.frame_batch_size<1: p.error('Workers must be nonnegative and frame batch size positive.')
    prepare(a.root,a.source_manifest,a.base_checkpoint,a.output,a.device,a.limit,
            decode_workers=a.decode_workers,frame_batch_size=a.frame_batch_size)

if __name__=='__main__': main()
