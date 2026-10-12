"""Evaluate caption queries against one combined dataset gallery."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from visual_retrieval.common import DEFAULT_ROOT, PROTOCOL, PREPROCESSING_V2, read_feature_bundle, write_json


from visual_retrieval.models.registry import validate_preprocessing

from .ranking_metrics import compute_ranks, compute_rankings, metrics_from_ranks
from .prediction_exports import save_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', type=Path,
                        default=Path(DEFAULT_ROOT) / 'features' / 'internvideo2_stage2_1b_all')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--top-k', type=int, default=10, help='Candidates per query saved for error analysis; ranks still use the full gallery.')
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Explicitly evaluate only successfully extracted videos.')
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error('--top-k must be positive.')
    report = evaluate_directory(args.features_dir, args.batch_size, args.allow_incomplete, args.top_k)
    print(json.dumps(report['text_to_video'], indent=2))
    print(f'Saved: {args.features_dir / "evaluation.json"}')
    print(f'Saved predictions: {args.features_dir / "query_ranks.csv"} and {args.features_dir / "predictions.jsonl"}')


def evaluate_directory(features_dir, batch_size=256, allow_incomplete=False, top_k=10):
    if not isinstance(top_k, (int, np.integer)) or top_k <= 0:
        raise ValueError('top_k must be a positive integer.')
    top_k = int(top_k)
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
    queries = json.loads((features_dir / 'queries.json').read_text(encoding='utf-8'))
    if len(queries) != len(bundle['text_features']) or [q['video_id'] for q in queries] != bundle['text_video_ids'].tolist():
        raise ValueError('queries.json is misaligned with feature labels.')
    rankings = compute_rankings(bundle['text_features'], bundle['video_features'],
                               [lookup[v] for v in bundle['text_video_ids']], batch_size, top_k)
    ranks = rankings['ranks']
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
              'prediction_export': {'schema_version': 1, 'top_k': rankings['top_indices'].shape[1],
                                    'requested_top_k': top_k, 'csv': 'query_ranks.csv', 'jsonl': 'predictions.jsonl',
                                    'label_type': 'gallery_item_id', 'score_type': 'cosine_similarity'},
              'created_utc': datetime.now(timezone.utc).isoformat()}
    save_predictions(features_dir, queries, video_ids, manifest, rankings)
    write_json(features_dir / 'evaluation.json', report)
    return report


if __name__ == '__main__':
    main()
