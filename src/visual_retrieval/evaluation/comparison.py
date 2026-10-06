"""Fair cross-model comparison; mismatched galleries/queries are rejected."""
import argparse
import csv
import json
from pathlib import Path
from visual_retrieval.common import read_feature_bundle, write_json
from visual_retrieval.config import canonical_dataset
from visual_retrieval.models.registry import validate_preprocessing
from .baseline_evaluation import evaluate_directory


def compare_bundles(directories, batch_size=256):
    if len(directories) < 2:
        raise ValueError('Supply at least two model feature directories.')
    reference = None
    models = set()
    for directory in map(Path, directories):
        bundle = read_feature_bundle(directory)
        manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        model = validate_preprocessing(manifest)
        if model in models:
            raise ValueError('Comparison requires distinct models.')
        models.add(model)
        ids = bundle['video_ids'].tolist()
        if set(ids) != set(manifest['requested_video_ids']) or manifest.get('complete') is False:
            raise ValueError(f'{directory}: incomplete gallery; retry extraction before comparison.')
        queries = json.loads((directory/'queries.json').read_text(encoding='utf-8'))
        if len(queries) != len(bundle['text_features']) or [q['video_id'] for q in queries] != bundle['text_video_ids'].tolist():
            raise ValueError(f'{directory}: queries and feature labels are misaligned.')
        items = {item['item_id']: item for item in manifest.get('gallery_items', [])}
        if manifest['protocol'] == 'combined_subset_activitynet_segments' and set(items) != set(ids):
            raise ValueError('Segment comparison requires gallery metadata with source videos and timestamps.')
        gallery_metadata = [(v, items.get(v, {}).get('source_video_id', v),
                             items.get(v, {}).get('start'), items.get(v, {}).get('end')) for v in ids]
        identity = (canonical_dataset(manifest.get('dataset', 'msrvtt')), manifest['protocol'], ids,
                    gallery_metadata,
                    [(q['query_id'], q['video_id'], q['text']) for q in queries])
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError('Models have different datasets, protocols, gallery order or queries. Use identical manifests.')
    reports = [evaluate_directory(Path(directory), batch_size) for directory in directories]
    return dict(dataset=reference[0], protocol=reference[1], num_items=len(reference[2]),
                num_queries=len(reference[4]), results=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dirs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    report = compare_bundles(args.features_dirs, args.batch_size)
    write_json(args.output, report)
    with args.output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset','model','gallery_items','queries','r1','r5','r10','mrr'])
        for result in report['results']:
            metrics=result['text_to_video']
            writer.writerow([report['dataset'],result['model_name'],report['num_items'],report['num_queries'],
                             *[metrics[key] for key in ('r1','r5','r10','mrr')]])
            print(result['model_name'],json.dumps(metrics))
    print(f'Saved comparison: {args.output}')


if __name__ == '__main__':
    main()
