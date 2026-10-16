"""CoVR-2 query scoring and token pooling, separate from cosine GalleryIndex."""
import numpy as np


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=-1) <= 1e-12):
        raise ValueError('Features must be finite and nonzero.')
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def score_video(query_tokens, frame_tokens, scoring_text, temperature=.1):
    query, frames, text = map(lambda x: np.asarray(x, dtype=np.float32),
                              (query_tokens, frame_tokens, scoring_text))
    if (query.ndim != 2 or frames.ndim != 3 or text.ndim != 1
            or frames.shape[1:] != query.shape or query.shape[-1] != text.shape[0]
            or not frames.shape[0] or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError('Expected query [Q,D], frames [F,Q,D], text [D].')
    query, frames, text = unit(query), unit(frames), unit(text)
    # BLIP-2 image/text scoring takes the maximum over image query tokens.
    relevance = np.max(frames @ text, axis=1)
    weights = np.exp((relevance - relevance.max()) / temperature)
    weights /= weights.sum()
    target = unit(np.einsum('f,fqd->qd', weights, frames))
    # Match upstream evaluation: normalize tokens, mean tokens, then dot product.
    return float(query.mean(axis=0) @ target.mean(axis=0))
