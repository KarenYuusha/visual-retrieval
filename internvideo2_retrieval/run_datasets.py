"""Extract/evaluate all three datasets using the same Python environment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from dataset_adapters import DATASETS, ROOT_NAMES
from retrieval_common import write_json


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', choices=DATASETS, nargs='+', default=list(DATASETS))
    parser.add_argument('--base-root', type=Path, default=Path(r'S:\video_retrieval'))
    for dataset in DATASETS:
        parser.add_argument('--' + dataset.replace('_', '-') + '-root', type=Path)
    parser.add_argument('--output-root', type=Path, help='Optional common output root; subfolders named by dataset.')
    parser.add_argument('--stage', choices=['all', 'check', 'extract', 'evaluate'], default='all')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--tokenizer', default='google-bert/bert-large-uncased')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--precision', choices=['auto', 'bf16', 'fp16', 'fp32'], default='auto')
    parser.add_argument('--text-batch-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=256, help='Evaluation query batch size.')
    parser.add_argument('--activitynet-mode', choices=['segments', 'video'], default='segments')
    parser.add_argument('--activitynet-val-file', choices=['val_1.json', 'val_2.json'], default='val_1.json')
    parser.add_argument('--allow-missing', action='store_true')
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--summary', type=Path, help='Optional path for aggregate metrics JSON.')
    return parser


def commands_for(args, dataset):
    scripts = Path(__file__).resolve().parent
    root = getattr(args, dataset + '_root') or args.base_root / ROOT_NAMES[dataset]
    output = args.output_root / dataset if args.output_root else root / 'features' / 'internvideo2_stage2_1b_all'
    extraction = [sys.executable, str(scripts / 'extract_features.py'), '--dataset', dataset,
                  '--data-root', str(root), '--output-dir', str(output),
                  '--device', args.device, '--precision', args.precision, '--tokenizer', args.tokenizer,
                  '--text-batch-size', str(args.text_batch_size)]
    if args.checkpoint:
        extraction.extend(['--checkpoint', str(args.checkpoint)])
    if args.allow_missing:
        extraction.append('--allow-missing')
    if dataset == 'activitynet_captions':
        extraction.extend(['--activitynet-mode', args.activitynet_mode, '--activitynet-val-file', args.activitynet_val_file])
    evaluation = [sys.executable, str(scripts / 'evaluate_retrieval.py'), '--features-dir', str(output),
                  '--batch-size', str(args.batch_size)]
    if args.allow_incomplete:
        evaluation.append('--allow-incomplete')
    return dict(check=extraction + ['--check-data'], extract=extraction, evaluate=evaluation, output=output)


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.text_batch_size <= 0 or args.batch_size <= 0:
        parser.error('Batch sizes must be positive.')
    jobs = [(dataset, commands_for(args, dataset)) for dataset in dict.fromkeys(args.datasets)]
    # Catch annotation/file errors across all requested datasets before downloading a checkpoint.
    if args.stage in ('all', 'check', 'extract'):
        for dataset, commands in jobs:
            print(f'\nChecking {dataset}', flush=True)
            subprocess.run(commands['check'], check=True)
    if args.stage == 'check':
        return
    reports = {}
    for dataset, commands in jobs:
        print(f'\nRunning {dataset}', flush=True)
        if args.stage in ('all', 'extract'):
            subprocess.run(commands['extract'], check=True)
        if args.stage in ('all', 'evaluate'):
            subprocess.run(commands['evaluate'], check=True)
            reports[dataset] = json.loads((commands['output'] / 'evaluation.json').read_text(encoding='utf-8'))
    if reports:
        path = args.summary or args.base_root / 'features' / 'internvideo2_summary.json'
        write_json(path, reports)
        print(f'\nSaved summary: {path}')
        for dataset, report in reports.items():
            print(dataset, json.dumps(report['text_to_video']))


if __name__ == '__main__':
    main()
