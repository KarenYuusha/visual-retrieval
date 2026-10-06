"""Atomic feature caching and provenance checks."""
import json
from pathlib import Path
from zipfile import BadZipFile
import numpy as np
from visual_retrieval.common import PREPROCESSING_V1, PREPROCESSING_V2, normalize_features, write_json

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
