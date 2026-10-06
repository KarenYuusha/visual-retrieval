"""Extract baseline embeddings with shared source decoding and bounded prefetch."""
import argparse
import json
import time
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from tqdm import tqdm

from visual_retrieval.data.adapters import DATASETS, ROOT_NAMES, load_dataset
from visual_retrieval.config import MODEL_NAMES, output_directory
from visual_retrieval.models.registry import MODEL_INFO, resolve_model_files, create_encoder, checkpoint_signature
from .pipeline import decode_source, prefetch_sources

from visual_retrieval.common import (PREPROCESSING_V1, PREPROCESSING_V2,
                              file_signature, normalize_features, sha256, write_json)


from .cache import atomic_npz, cached_video, ensure_cache_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODEL_NAMES, default='internvideo2')
    parser.add_argument('--dataset', choices=DATASETS, default='msrvtt')
    parser.add_argument('--data-root', type=Path, help='Dataset root with subset.json and raw_videos.')
    parser.add_argument('--annotations', type=Path, nargs='+')
    parser.add_argument('--activitynet-mode', choices=['segments', 'video'], default='segments')
    parser.add_argument('--activitynet-val-file', choices=['val_1.json', 'val_2.json'], default='val_1.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='InternVideo2 .pt, trained CLIP4Clip .bin/.pt, or local CLIP HF directory.')
    parser.add_argument('--tokenizer',
                        help='Hugging Face tokenizer ID or local tokenizer directory.')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--precision', default='auto', choices=['auto', 'bf16', 'fp16', 'fp32'])
    parser.add_argument('--text-batch-size', type=int, default=32)
    parser.add_argument('--decode-workers', type=int, default=2, help='Source decoding threads; 0 disables prefetch.')
    parser.add_argument('--video-batch-size', type=int, help='Gallery items per inference batch; default 4 for CLIP/CLIP4Clip, 1 for InternVideo2.')
    parser.add_argument('--check-data', action='store_true', help='Check IDs, files and captions without loading a model.')
    parser.add_argument('--allow-missing', action='store_true', help='Continue extraction if video files are missing.')
    from visual_retrieval.cli.configuration import parse_configured
    args = parse_configured(parser)
    if args.text_batch_size <= 0:
        parser.error('--text-batch-size must be positive.')
    if args.decode_workers < 0 or (args.video_batch_size is not None and args.video_batch_size <= 0):
        parser.error('--decode-workers must be nonnegative and --video-batch-size must be positive.')
    video_batch_size = args.video_batch_size or (1 if args.model == 'internvideo2' else 4)
    args.data_root = args.data_root or Path(r'S:\video_retrieval') / ROOT_NAMES[args.dataset]
    if args.dataset == 'activitynet':
        args.dataset = 'activitynet_captions'
    subset_path = args.data_root / 'subset.json'
    dataset = load_dataset(args.dataset, args.data_root, args.annotations,
                           args.activitynet_mode, args.activitynet_val_file)
    items = dataset.items
    ids = [item['item_id'] for item in items]
    all_queries = dataset.queries
    missing = [item['item_id'] for item in items if not item['path'].is_file()]
    gallery_unit = 'gallery segments' if args.dataset == 'activitynet_captions' and args.activitynet_mode == 'segments' else 'unique videos'
    print(f'Combined subset: {len(ids)} {gallery_unit}; {len(all_queries)} caption queries.')
    print(f'Source videos: {len({item["source_video_id"] for item in items})}')
    print(f'Dataset: {args.dataset}; splits: {"+".join(dataset.splits)}; protocol: {dataset.protocol}')
    print(f'Annotations: {dataset.annotation_paths}; missing gallery files: {len(missing)}')
    if missing:
        print('Missing IDs:', ', '.join(missing[:20]))
        if not args.allow_missing:
            raise FileNotFoundError('Restore missing files before extraction, or use --allow-missing explicitly.')
    if args.check_data:
        print('Data check passed. No model loaded and no features extracted.')
        return

    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install CUDA-enabled PyTorch before extraction.')
    if args.device == 'cpu' and args.precision not in ('auto', 'fp32'):
        raise ValueError('CPU inference requires fp32.')
    extraction_started = time.perf_counter()
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    checkpoint, args.tokenizer = resolve_model_files(args.model, args.checkpoint, args.tokenizer)
    info = MODEL_INFO[args.model]
    output = args.output_dir or output_directory(args.model, args.data_root)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'video_cache'
    cache.mkdir(exist_ok=True)
    run_config = {'model_id': info['model_id'], 'source_commit': info['source_commit'],
                  'protocol': dataset.protocol, 'subset_sha256': sha256(subset_path),
                  'annotations_sha256': sha256(dataset.annotation_paths[0]) if len(dataset.annotation_paths) == 1
                  else [sha256(path) for path in dataset.annotation_paths],
                  'checkpoint': checkpoint_signature(checkpoint), 'tokenizer': args.tokenizer,
                  'precision': args.precision, 'device': args.device,
                  'preprocessing': info['preprocessing']}
    if args.model != 'internvideo2':
        run_config['model_name'] = args.model
    if args.dataset != 'msrvtt':
        run_config.update(dataset=args.dataset, activitynet_mode=args.activitynet_mode if args.dataset == 'activitynet_captions' else None)
    ensure_cache_config(output, run_config)

    encoder = None
    effective_precision = None

    def ensure_model():
        nonlocal encoder, effective_precision
        if encoder is None:
            encoder = create_encoder(args.model, checkpoint, args.device, args.precision, args.tokenizer)
            effective_precision = encoder.effective_precision
        return encoder

    features_by_id, failed, sources = {}, [], {}
    reused = 0
    # Validate caches before queuing work: fully cached sources are never decoded.
    for item in items:
        video_id = item['item_id']
        video_path = item['path']
        if not video_path.is_file():
            failed.append({'video_id': video_id, 'error': 'file_missing'})
            continue
        signature = file_signature(video_path)
        if item['start'] is not None:
            signature.update(start=item['start'], end=item['end'])
        cache_path = cache / f'{video_id}.npz'
        feature = cached_video(cache_path, signature)
        if feature is not None:
            reused += 1
            features_by_id[video_id] = feature
        else:
            sources.setdefault(str(video_path.resolve()), []).append({**item, 'signature': signature})
    print(f'Decoding {len(sources)} uncached sources with {args.decode_workers} workers; '
          f'video batch size {video_batch_size}; reused items {reused}.')
    inference_batches = 0
    def encode_batch(batch):
        nonlocal inference_batches
        # Model failures abort; all previously committed item caches are resumable.
        features = normalize_features(ensure_model().encode_videos([result[0] for _, result in batch]))
        if features.shape != (len(batch), 512):
            raise ValueError(f'Unexpected batched embedding shape: {features.shape}')
        for (item, (_, indices, count)), feature in zip(batch, features):
            video_id = item['item_id']
            atomic_npz(cache / f'{video_id}.npz', feature=feature, source_signature=json.dumps(item['signature']),
                       frame_indices=np.array(indices), decoded_frames=count)
            features_by_id[video_id] = feature
        inference_batches += 1

    stream = prefetch_sources(sources.values(), lambda group: decode_source(group, info['num_frames']), args.decode_workers)
    try:
        with tqdm(total=len(items), initial=reused + len(failed), desc='Gallery', unit='item') as progress:
            batch = []
            for group in stream:
                for item, result in group:
                    if isinstance(result, Exception):
                        failed.append({'video_id': item['item_id'], 'error': str(result)})
                        progress.update(1)
                        continue
                    batch.append((item, result))
                    if len(batch) == video_batch_size:
                        encode_batch(batch)
                        progress.update(len(batch))
                        batch.clear()
            if batch:
                encode_batch(batch)
                progress.update(len(batch))
    finally:
        stream.close()
    # Grouping/caching must not change gallery order (including stable ranking ties).
    successful_ids = [video_id for video_id in ids if video_id in features_by_id]
    video_features = [features_by_id[video_id] for video_id in successful_ids]
    write_json(output / 'failed_videos.json', failed)
    if not successful_ids:
        raise ValueError('No videos were extracted. See failed_videos.json.')

    text_cache_path = output / 'text_cache.npz'
    text_features = None
    labels = np.array([q['video_id'] for q in all_queries])
    if text_cache_path.is_file():
        try:
            with np.load(text_cache_path, allow_pickle=False) as stored:
                candidate = normalize_features(stored['text_features'])
                if (candidate.shape == (len(all_queries), 512)
                        and np.array_equal(stored['text_video_ids'], labels)):
                    text_features = candidate
        except (ValueError, KeyError, OSError, BadZipFile, EOFError):
            pass
    if text_features is None:
        active_encoder = ensure_model()
        text_features = np.empty((len(all_queries), 512), dtype=np.float32)
        for start in tqdm(range(0, len(all_queries), args.text_batch_size), desc='Text batches'):
            batch = all_queries[start:start + args.text_batch_size]
            text_features[start:start + len(batch)] = active_encoder.encode_text([q['text'] for q in batch])
        text_features = normalize_features(text_features)
        atomic_npz(text_cache_path, text_features=text_features, text_video_ids=labels)

    successful = set(successful_ids)
    keep = np.array([q['video_id'] in successful for q in all_queries])
    queries = [q for q, include in zip(all_queries, keep) if include]
    atomic_npz(output / 'features.npz', video_features=np.stack(video_features).astype(np.float32),
               video_ids=np.array(successful_ids), text_features=text_features[keep],
               text_video_ids=labels[keep])
    # Existing metrics must never describe a newly written feature bundle.
    for name in ('evaluation.json', 'query_ranks.csv'):
        (output / name).unlink(missing_ok=True)
    write_json(output / 'queries.json', queries)
    write_json(output / 'manifest.json', {
        **run_config, 'dataset': args.dataset, 'splits': dataset.splits,
        'gallery_items': [{**item, 'path': str(item['path'].resolve())} for item in items],
        'requested_video_ids': ids, 'successful_video_ids': successful_ids,
        'failed_videos': failed, 'num_queries': len(queries), 'reused_video_embeddings': reused,
        'effective_precision': effective_precision, 'complete': not failed,
        'efficiency': {'extraction_wall_seconds': time.perf_counter() - extraction_started,
                       'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                       'feature_bundle_bytes': (output / 'features.npz').stat().st_size,
                       'decode_workers': args.decode_workers, 'video_batch_size': video_batch_size,
                       'decoded_source_videos': len(sources), 'video_inference_batches': inference_batches,
                       'cached_video_items': reused},
    })
    print(f'Saved {len(successful_ids)} video features and {len(queries)} text features to {output}')
    print(f'Reused: {reused}; failed: {len(failed)}. Rerun this command to retry failures.')
    if failed:
        raise SystemExit('Extraction is incomplete. See failed_videos.json; default evaluation will refuse it.')


if __name__ == '__main__':
    main()
