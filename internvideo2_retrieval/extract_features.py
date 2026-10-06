"""Extract InternVideo2 embeddings for a combined dataset subset."""
import argparse
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from tqdm import tqdm

from dataset_adapters import DATASETS, ROOT_NAMES, load_dataset

from retrieval_common import (PREPROCESSING_V1, PREPROCESSING_V2,
                              file_signature, normalize_features, sha256, write_json)


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as handle:
        np.savez(handle, **arrays)
    temp.replace(path)


def cached_video(path, signature):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data['source_signature'])) != signature:
                return None
            features = normalize_features(data['feature'].reshape(1, -1))[0]
            if features.shape != (512,):
                return None
            return features
    except (ValueError, KeyError, OSError, BadZipFile, EOFError):
        return None


def ensure_cache_config(output, run_config):
    output = Path(output)
    config_path = output / 'run_config.json'
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding='utf-8'))
        legacy_config = {**run_config, 'preprocessing': PREPROCESSING_V1}
        if run_config.get('preprocessing') == PREPROCESSING_V2 and existing == legacy_config:
            # Only tokenization changed: safely reuse videos from identical
            # weights/data/settings, but never reuse text embeddings from v1.
            for name in ('evaluation.json', 'query_ranks.csv'):
                old = output / name
                backup = output / f'{old.stem}_before_tokenizer_fix{old.suffix}'
                if old.is_file() and not backup.exists():
                    old.replace(backup)
            for name in ('text_cache.npz', 'features.npz', 'queries.json', 'manifest.json',
                         'evaluation.json', 'query_ranks.csv'):
                (output / name).unlink(missing_ok=True)
            write_json(config_path, run_config)
            print('Tokenizer correction detected: retaining video caches and regenerating text embeddings.')
            return
        if existing != run_config:
            raise ValueError('These cached features were created with different settings/data/weights. '
                             'Use a new --output-dir to avoid mixing embeddings.')
        return
    if (any((output / 'video_cache').glob('*.npz'))
            or (output / 'text_cache.npz').exists() or (output / 'features.npz').exists()):
        raise ValueError('Found orphan caches without run_config.json. Use a fresh --output-dir; '
                         'their model/settings provenance cannot be verified.')
    write_json(config_path, run_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, default='msrvtt')
    parser.add_argument('--data-root', type=Path, help='Dataset root with subset.json and raw_videos.')
    parser.add_argument('--annotations', type=Path, nargs='+')
    parser.add_argument('--activitynet-mode', choices=['segments', 'video'], default='segments')
    parser.add_argument('--activitynet-val-file', choices=['val_1.json', 'val_2.json'], default='val_1.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='Full Stage2 .pt file; auto-download if omitted.')
    parser.add_argument('--tokenizer', default='google-bert/bert-large-uncased',
                        help='Hugging Face tokenizer ID or local tokenizer directory.')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--precision', default='auto', choices=['auto', 'bf16', 'fp16', 'fp32'])
    parser.add_argument('--text-batch-size', type=int, default=32)
    parser.add_argument('--check-data', action='store_true', help='Check IDs, files and captions without loading a model.')
    parser.add_argument('--allow-missing', action='store_true', help='Continue extraction if video files are missing.')
    args = parser.parse_args()
    if args.text_batch_size <= 0:
        parser.error('--text-batch-size must be positive.')
    args.data_root = args.data_root or Path(r'S:\video_retrieval') / ROOT_NAMES[args.dataset]
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
    from iv2_model import (MODEL_ID, SOURCE_COMMIT, encode_text_batch, load_model,
                          prepare_frames, read_video_frames, resolve_checkpoint)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install CUDA-enabled PyTorch before extraction.')
    checkpoint = resolve_checkpoint(args.checkpoint)
    output = args.output_dir or args.data_root / 'features' / 'internvideo2_stage2_1b_all'
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'video_cache'
    cache.mkdir(exist_ok=True)
    run_config = {'model_id': MODEL_ID, 'source_commit': SOURCE_COMMIT,
                  'protocol': dataset.protocol, 'subset_sha256': sha256(subset_path),
                  'annotations_sha256': sha256(dataset.annotation_paths[0]) if len(dataset.annotation_paths) == 1
                  else [sha256(path) for path in dataset.annotation_paths],
                  'checkpoint': file_signature(checkpoint), 'tokenizer': args.tokenizer,
                  'precision': args.precision, 'device': args.device,
                  'preprocessing': PREPROCESSING_V2}
    if args.dataset != 'msrvtt':
        run_config.update(dataset=args.dataset, activitynet_mode=args.activitynet_mode if args.dataset == 'activitynet_captions' else None)
    ensure_cache_config(output, run_config)

    model = tokenizer = None
    effective_precision = None

    def ensure_model():
        nonlocal model, tokenizer, effective_precision
        if model is None:
            model, tokenizer, effective_precision = load_model(
                checkpoint, args.device, args.precision, args.tokenizer)
        return model, tokenizer

    video_features, successful_ids, failed = [], [], []
    reused = 0
    for item in tqdm(items, desc='Gallery', unit='item'):
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
        else:
            try:
                rgb, indices, count = read_video_frames(video_path, item['start'], item['end'])
            except (ValueError, OSError) as error:
                failed.append({'video_id': video_id, 'error': str(error)})
                continue
            active_model, _ = ensure_model()
            # Batch size 1 keeps memory usage modest. Model errors (e.g. OOM)
            # abort immediately; previously written video caches remain resumable.
            feature = active_model.video_features(prepare_frames(rgb).to(args.device)).cpu().numpy().reshape(-1)
            feature = normalize_features(feature.reshape(1, -1))[0]
            if feature.shape != (512,):
                raise ValueError(f'Unexpected embedding shape for {video_id}: {feature.shape}')
            atomic_npz(cache_path, feature=feature, source_signature=json.dumps(signature),
                       frame_indices=np.array(indices), decoded_frames=count)
        video_features.append(feature)
        successful_ids.append(video_id)
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
        active_model, active_tokenizer = ensure_model()
        text_features = np.empty((len(all_queries), 512), dtype=np.float32)
        for start in tqdm(range(0, len(all_queries), args.text_batch_size), desc='Text batches'):
            batch = all_queries[start:start + args.text_batch_size]
            text_features[start:start + len(batch)] = encode_text_batch(
                active_model, active_tokenizer, [q['text'] for q in batch], args.device)
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
    })
    print(f'Saved {len(successful_ids)} video features and {len(queries)} text features to {output}')
    print(f'Reused: {reused}; failed: {len(failed)}. Rerun this command to retry failures.')
    if failed:
        raise SystemExit('Extraction is incomplete. See failed_videos.json; default evaluation will refuse it.')


if __name__ == '__main__':
    main()
