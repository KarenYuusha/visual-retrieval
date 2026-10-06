"""Normalized query update from positive and negative gallery feedback."""
import numpy as np
from visual_retrieval.common import normalize_features


def refine_vector(query, positive=None, negative=None, beta=.5, gamma=.25):
    if not np.isfinite(beta) or not np.isfinite(gamma) or beta < 0 or gamma < 0:
        raise ValueError('Feedback weights must be finite and nonnegative.')
    vector = normalize_features(np.asarray(query,dtype=np.float32).reshape(1,-1))[0]
    for features, weight in ((positive,beta),(negative,-gamma)):
        if features is None or len(features) == 0:
            continue
        features = normalize_features(features)
        if features.shape[1] != len(vector):
            raise ValueError('Feedback dimensions differ from the active model query.')
        vector = vector + weight * features.mean(axis=0)
    return normalize_features(vector.reshape(1,-1))[0]
