import numpy as np
import pytest


def test_scoring_matches_upstream_token_normalization_and_weighted_pooling():
    from visual_retrieval.composed.scoring import score_video
    frames = np.array([[[1.,0.],[0.,1.]], [[0.,1.],[0.,1.]]], np.float32)
    query = np.array([[1.,0.],[1.,0.]], np.float32)
    text = np.array([1.,0.], np.float32)
    # LAVIS query scoring: best image query-token similarity to the text CLS.
    frame_scores = np.max(frames @ text, axis=1)
    weights = np.exp((frame_scores - frame_scores.max()) / .1)
    weights /= weights.sum()
    pooled = (frames * weights[:,None,None]).sum(axis=0)
    pooled /= np.linalg.norm(pooled, axis=-1, keepdims=True)
    expected = query.mean(0) @ pooled.mean(0)
    assert score_video(query, frames, text) == pytest.approx(expected, abs=1e-6)
    # Final token means must NOT be normalized again (would give cosine=1).
    assert score_video(query, frames, text) < .8


def test_invalid_or_nonfinite_frame_features_are_rejected():
    from visual_retrieval.composed.scoring import score_video
    with pytest.raises(ValueError):
        score_video(np.ones((2,3)), np.ones((4,2,4)), np.ones(3))
    with pytest.raises(ValueError):
        score_video(np.ones((2,3)), np.full((4,2,3), np.nan), np.ones(3))
