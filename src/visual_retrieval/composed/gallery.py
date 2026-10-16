"""Resumable per-video token caches and exact CoVR-2 ranking."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .scoring import score_video

FORMAT = 'covr2_blip2_coco_rgb_square364_15centers_token_l2_v1'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_tokens(tokens):
    if (not isinstance(tokens,np.ndarray) or tokens.dtype not in (np.dtype('float16'),np.dtype('float32'))
            or tokens.shape != (15,32,256) or not np.isfinite(tokens).all()):
        raise ValueError('Invalid CoVR-2 gallery: expected finite [15,32,256] tokens.')
    norms = np.linalg.norm(tokens.astype(np.float32), axis=-1)
    if not np.allclose(norms, 1, atol=.005):
        raise ValueError('CoVR-2 gallery tokens must be normalized.')
    return tokens


def token_path(folder, item_id):
    return Path(folder) / 'tokens' / (hashlib.sha256(item_id.encode()).hexdigest() + '.npy')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def rank_gallery(query, text, items, source=None, limit=100):
    scored = []
    for row, frames in items:
        if source is not None and (row['dataset'], row['source_video_id']) == tuple(source):
            continue
        scored.append(dict(row, cosine_similarity=score_video(query, frames, text), model='covr2'))
    scored.sort(key=lambda row: (-row['cosine_similarity'], row['dataset'], row['item_id']))
    return [dict(row, rank=i+1) for i, row in enumerate(scored[:limit])]
