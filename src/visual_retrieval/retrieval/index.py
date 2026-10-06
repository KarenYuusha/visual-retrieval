"""Exact cosine retrieval for class-project galleries."""
import numpy as np
from visual_retrieval.common import normalize_features, read_feature_bundle


class GalleryIndex:
    def __init__(self, item_ids, features):
        self.item_ids = list(item_ids)
        self.features = normalize_features(features)
        if len(self.item_ids) != len(self.features) or len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError('Gallery IDs must be unique and match feature rows.')
        self.lookup = {key: i for i, key in enumerate(self.item_ids)}

    @classmethod
    def from_directory(cls, directory):
        bundle = read_feature_bundle(directory)
        return cls(bundle['video_ids'].tolist(), bundle['video_features'])

    def vectors(self, item_ids):
        return self.features[[self.lookup[key] for key in item_ids]]

    def search(self, query, top_k=10):
        query = np.asarray(query,dtype=np.float32).reshape(1,-1)
        if top_k <= 0 or query.shape[1] != self.features.shape[1]:
            raise ValueError('Invalid top-k or query dimension.')
        query = normalize_features(query)
        scores = (query @ self.features.T)[0]
        order = np.argsort(-scores,kind='stable')[:top_k]
        return [dict(rank=rank,item_id=self.item_ids[i],cosine_similarity=float(scores[i]))
                for rank,i in enumerate(order,1)]
