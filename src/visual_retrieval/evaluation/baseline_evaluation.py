"""Evaluate caption queries against one combined dataset gallery."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from visual_retrieval.common import DEFAULT_ROOT, PROTOCOL, PREPROCESSING_V2, read_feature_bundle, write_json


from visual_retrieval.models.registry import validate_preprocessing

from .ranking_metrics import compute_ranks, metrics_from_ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', type=Path,
                        default=Path(DEFAULT_ROOT) / 'features' / 'internvideo2_stage2_1b_all')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Explicitly evaluate only successfully extracted videos.')
    args = parser.parse_args()
    report = evaluate_directory(args.features_dir, args.batch_size, args.allow_incomplete)
    print(json.dumps(report['text_to_video'], indent=2))
    print(f'Saved: {args.features_dir / "evaluation.json"}')


def evaluate_directory(features_dir, batch_size=256, allow_incomplete=False):
    features_dir = Path(features_dir)
    bundle = read_feature_bundle(features_dir)
    manifest = json.loads((features_dir / 'manifest.json').read_text(encoding='utf-8'))
    model_name = validate_preprocessing(manifest)
    video_ids = bundle['video_ids'].tolist()
    requested = manifest['requested_video_ids']
    omitted = [v for v in requested if v not in set(video_ids)]
    if omitted and not allow_incomplete:
        raise ValueError(f'{len(omitted)} requested videos are missing. Rerun extraction to retry '
                         'them, or explicitly use --allow-incomplete.')
    if manifest.get('protocol') not in (PROTOCOL, 'combined_subset_all_english_captions',
                                        'combined_subset_activitynet_segments', 'combined_subset_activitynet_video') or set(video_ids) - set(requested):
        raise ValueError('Feature manifest does not describe the expected combined gallery.')
    lookup = {video_id: i for i, video_id in enumerate(video_ids)}
    ranks = compute_ranks(bundle['text_features'], bundle['video_features'],
                          [lookup[v] for v in bundle['text_video_ids']], batch_size)
    queries = json.loads((features_dir / 'queries.json').read_text(encoding='utf-8'))
    if len(queries) != len(ranks) or [q['video_id'] for q in queries] != bundle['text_video_ids'].tolist():
        raise ValueError('queries.json is misaligned with feature labels.')
    report = {'model': manifest['model_id'] if 'model_id' in manifest else 'InternVideo2-Stage2_1B-224p-f4', 'model_name': model_name, 'dataset': manifest.get('dataset', 'msrvtt'),
              'gallery': '+'.join(manifest.get('splits', ['train', 'test'])),
              'protocol': manifest['protocol'], 'preprocessing': manifest['preprocessing'],
              'retrieval': 'cosine_similarity_without_reranking',
              'efficiency': manifest.get('efficiency'),
              'num_requested_videos': len(requested), 'num_videos': len(video_ids),
              'num_queries': len(ranks),
              'num_source_videos': len({i['source_video_id'] for i in manifest.get('gallery_items', []) if i['item_id'] in lookup}) or len(video_ids),
              'complete': not omitted, 'omitted_video_ids': omitted,
              'units': {'r1': 'percent', 'r5': 'percent', 'r10': 'percent', 'mrr': '0_to_1'},
              'tie_policy': 'gallery_order', 'text_to_video': metrics_from_ranks(ranks),
              'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(features_dir / 'evaluation.json', report)
    with (features_dir / 'query_ranks.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['query_id', 'video_id', 'caption', 'rank', 'reciprocal_rank'])
        for query, rank in zip(queries, ranks):
            writer.writerow([query['query_id'], query['video_id'], query['text'], int(rank), 1 / rank])
    return report


if __name__ == '__main__':
    main()
