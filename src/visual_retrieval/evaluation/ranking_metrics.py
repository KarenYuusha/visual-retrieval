"""Full-rank retrieval metrics and deterministic gallery-order ties."""
import numpy as np


def compute_ranks(text_features, video_features, target_indices, batch_size=256):
    return compute_rankings(text_features, video_features, target_indices, batch_size, top_k=0)['ranks']


def compute_rankings(text_features, video_features, target_indices, batch_size=256, top_k=10):
    """Full target ranks and stable top candidates from the same score batches."""
    text = np.asarray(text_features, dtype=np.float32)
    video = np.asarray(video_features, dtype=np.float32)
    targets = np.asarray(target_indices, dtype=np.int64)
    if (text.ndim != 2 or video.ndim != 2 or text.shape[1] != video.shape[1]
            or targets.shape != (len(text),) or not len(text) or not len(video)
            or batch_size <= 0 or (targets < 0).any() or (targets >= len(video)).any()
            or not np.isfinite(text).all() or not np.isfinite(video).all()):
        raise ValueError('Invalid features, targets, or batch size.')
    if not isinstance(top_k, (int, np.integer)) or top_k < 0:
        raise ValueError('top_k must be a nonnegative integer.')
    top_k = min(top_k, len(video))
    ranks = np.empty(len(text), dtype=np.int64)
    target_scores = np.empty(len(text), dtype=np.float32)
    top_indices = np.empty((len(text), top_k), dtype=np.int64)
    top_scores = np.empty((len(text), top_k), dtype=np.float32)
    gallery_indices = np.arange(len(video))
    for start in range(0, len(text), batch_size):
        scores = text[start:start + batch_size] @ video.T
        gt = targets[start:start + batch_size]
        positive = scores[np.arange(len(scores)), gt][:, None]
        target_scores[start:start + len(scores)] = positive[:, 0]
        # Deterministic tie policy: earlier gallery ID wins. Rank is 1-based.
        ranks[start:start + len(scores)] = 1 + (scores > positive).sum(axis=1) + (
            (scores == positive) & (gallery_indices[None, :] < gt[:, None])).sum(axis=1)
        if top_k:
            indices = np.argsort(-scores, axis=1, kind='stable')[:, :top_k]
            top_indices[start:start + len(scores)] = indices
            top_scores[start:start + len(scores)] = np.take_along_axis(scores, indices, axis=1)
    return dict(ranks=ranks, target_scores=target_scores, top_indices=top_indices, top_scores=top_scores)


def metrics_from_ranks(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)
    if ranks.ndim != 1 or not len(ranks) or (ranks < 1).any():
        raise ValueError('Expected nonempty 1-based ranks.')
    return {'r1': float(100 * np.mean(ranks <= 1)),
            'r5': float(100 * np.mean(ranks <= 5)),
            'r10': float(100 * np.mean(ranks <= 10)),
            'mrr': float(np.mean(1.0 / ranks))}
