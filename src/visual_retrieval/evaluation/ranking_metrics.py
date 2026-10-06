"""Full-rank retrieval metrics and deterministic gallery-order ties."""
import numpy as np


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
