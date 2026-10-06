"""Evaluate caption queries against one combined dataset gallery."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from retrieval_common import DEFAULT_ROOT, PROTOCOL, PREPROCESSING_V2, read_feature_bundle, write_json


def compute_ranks(text_features, video_features, target_indices, batch_size=256):
    text = np.asarray(text_features, dtype=np.float32)
    video = np.asarray(video_features, dtype=np.float32)
    targets = np.asarray(target_indices, dtype=np.int64)
    if (text.ndim != 2 or video.ndim != 2 or text.shape[1] != video.shape[1]
            or targets.shape != (len(text),) or not len(text) or not len(video)
            or batch_size <= 0 or (targets < 0).any() or (targets >= len(video)).any()
            or not np.isfinite(text).all() or not np.isfinite(video).all()):
        raise ValueError('Invalid features, targets, or batch size.')
    ranks = np.empty(len(text), dtype=np.int64)
    gallery_indices = np.arange(len(video))
    for start in range(0, len(text), batch_size):
        scores = text[start:start + batch_size] @ video.T
        gt = targets[start:start + batch_size]
        positive = scores[np.arange(len(scores)), gt][:, None]
        # Deterministic tie policy: earlier gallery ID wins. Rank is 1-based.
        ranks[start:start + len(scores)] = 1 + (scores > positive).sum(axis=1) + (
            (scores == positive) & (gallery_indices[None, :] < gt[:, None])).sum(axis=1)
    return ranks


def metrics_from_ranks(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)
    if ranks.ndim != 1 or not len(ranks) or (ranks < 1).any():
        raise ValueError('Expected nonempty 1-based ranks.')
    return {'r1': float(100 * np.mean(ranks <= 1)),
            'r5': float(100 * np.mean(ranks <= 5)),
            'r10': float(100 * np.mean(ranks <= 10)),
            'mrr': float(np.mean(1.0 / ranks))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', type=Path,
                        default=Path(DEFAULT_ROOT) / 'features' / 'internvideo2_stage2_1b_all')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Explicitly evaluate only successfully extracted videos.')
    args = parser.parse_args()
    bundle = read_feature_bundle(args.features_dir)
    manifest = json.loads((args.features_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('preprocessing') != PREPROCESSING_V2:
        raise ValueError('These embeddings predate the tokenizer correction. Rerun extract_features.py '
                         'with the updated code to regenerate text embeddings before evaluation.')
    video_ids = bundle['video_ids'].tolist()
    requested = manifest['requested_video_ids']
    omitted = [v for v in requested if v not in set(video_ids)]
    if omitted and not args.allow_incomplete:
        raise ValueError(f'{len(omitted)} requested videos are missing. Rerun extraction to retry '
                         'them, or explicitly use --allow-incomplete.')
    if manifest.get('protocol') not in (PROTOCOL, 'combined_subset_all_english_captions',
                                        'combined_subset_activitynet_segments', 'combined_subset_activitynet_video') or set(video_ids) - set(requested):
        raise ValueError('Feature manifest does not describe the expected combined gallery.')
    lookup = {video_id: i for i, video_id in enumerate(video_ids)}
    ranks = compute_ranks(bundle['text_features'], bundle['video_features'],
                          [lookup[v] for v in bundle['text_video_ids']], args.batch_size)
    queries = json.loads((args.features_dir / 'queries.json').read_text(encoding='utf-8'))
    if len(queries) != len(ranks) or [q['video_id'] for q in queries] != bundle['text_video_ids'].tolist():
        raise ValueError('queries.json is misaligned with feature labels.')
    report = {'model': 'InternVideo2-Stage2_1B-224p-f4', 'dataset': manifest.get('dataset', 'msrvtt'),
              'gallery': '+'.join(manifest.get('splits', ['train', 'test'])),
              'protocol': manifest['protocol'], 'preprocessing': PREPROCESSING_V2,
              'retrieval': 'cosine_similarity_without_reranking',
              'num_requested_videos': len(requested), 'num_videos': len(video_ids),
              'num_queries': len(ranks),
              'num_source_videos': len({i['source_video_id'] for i in manifest.get('gallery_items', []) if i['item_id'] in lookup}) or len(video_ids),
              'complete': not omitted, 'omitted_video_ids': omitted,
              'units': {'r1': 'percent', 'r5': 'percent', 'r10': 'percent', 'mrr': '0_to_1'},
              'tie_policy': 'gallery_order', 'text_to_video': metrics_from_ranks(ranks),
              'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(args.features_dir / 'evaluation.json', report)
    with (args.features_dir / 'query_ranks.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['query_id', 'video_id', 'caption', 'rank', 'reciprocal_rank'])
        for query, rank in zip(queries, ranks):
            writer.writerow([query['query_id'], query['video_id'], query['text'], int(rank), 1 / rank])
    print(f'Gallery: {len(video_ids)} items ({report["gallery"]}); queries: {len(ranks)} captions')
    print(json.dumps(report['text_to_video'], indent=2))
    print(f'Saved: {args.features_dir / "evaluation.json"}')


if __name__ == '__main__':
    main()
