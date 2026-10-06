"""Retrieve videos for a new English text query using saved gallery embeddings."""
import argparse
import json
from pathlib import Path

import numpy as np

from retrieval_common import DEFAULT_ROOT, PREPROCESSING_V2, read_feature_bundle, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', required=True)
    parser.add_argument('--features-dir', type=Path,
                        default=Path(DEFAULT_ROOT) / 'features' / 'internvideo2_stage2_1b_all')
    parser.add_argument('--data-root', type=Path, help='Override saved source directory for relocated videos.')
    parser.add_argument('--checkpoint', type=Path, help='Override stored checkpoint location.')
    parser.add_argument('--tokenizer', help='Override tokenizer location with the same tokenizer vocabulary.')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--output', type=Path, help='Optional JSON results file.')
    args = parser.parse_args()
    if not args.query.strip() or args.top_k <= 0:
        parser.error('Supply a nonempty query and positive --top-k.')
    config = json.loads((args.features_dir / 'run_config.json').read_text(encoding='utf-8'))
    if config.get('preprocessing') != PREPROCESSING_V2:
        raise ValueError('Saved text embeddings use the earlier tokenizer. Rerun extract_features.py '
                         'with the updated code; it will retain video caches and refresh text embeddings.')
    from iv2_model import encode_text_batch, load_model, resolve_checkpoint
    checkpoint = resolve_checkpoint(args.checkpoint or config['checkpoint']['path'])
    # Check identity before using an alternate checkpoint to encode a query.
    stat = checkpoint.stat()
    if stat.st_size != config['checkpoint']['size'] or stat.st_mtime_ns != config['checkpoint']['mtime_ns']:
        raise ValueError('Checkpoint signature differs from the extraction checkpoint. '
                         'Use the original checkpoint or extract a new gallery.')
    model, tokenizer, _ = load_model(checkpoint, args.device,
                                    config['precision'] if args.device != 'cpu' else 'fp32',
                                    args.tokenizer or config['tokenizer'])
    bundle = read_feature_bundle(args.features_dir)
    query = encode_text_batch(model, tokenizer, [args.query], args.device)
    scores = (query @ bundle['video_features'].T)[0]
    order = np.argsort(-scores, kind='stable')[:args.top_k]
    manifest = json.loads((args.features_dir / 'manifest.json').read_text(encoding='utf-8'))
    items = {i['item_id']: i for i in manifest.get('gallery_items', [])}
    results = []
    for rank, index in enumerate(order, 1):
        video_id = str(bundle['video_ids'][index])
        item = items.get(video_id, {'source_video_id': video_id, 'start': None, 'end': None,
                                    'path': str(Path(DEFAULT_ROOT) / 'raw_videos' / f'{video_id}.mp4')})
        path = args.data_root / 'raw_videos' / Path(item['path']).name if args.data_root else item['path']
        result = {'rank': rank, 'video_id': video_id, 'cosine_similarity': float(scores[index]),
                  'path': str(path), 'source_video_id': item['source_video_id'],
                  'start': item['start'], 'end': item['end']}
        results.append(result)
        print(f'{rank:2d}. {video_id:12s} cosine={scores[index]:.4f}  {result["path"]}')
    if args.output:
        write_json(args.output, {'query': args.query, 'results': results})


if __name__ == '__main__':
    main()
