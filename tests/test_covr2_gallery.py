import numpy as np
import pytest
from visual_retrieval.composed.gallery import rank_gallery, validate_tokens


def test_ranking_excludes_known_source_and_keeps_dataset_identity():
    x = np.eye(2, dtype=np.float32)
    rows = [dict(dataset='a', item_id='same', source_video_id='s'),
            dict(dataset='b', item_id='same', source_video_id='s'),
            dict(dataset='a', item_id='other', source_video_id='t')]
    frames = [x[None]] * 3
    result = rank_gallery(x, np.ones(2), zip(rows, frames), source=('a','s'), limit=100)
    assert [(r['dataset'], r['item_id']) for r in result] == [('a','other'),('b','same')]
    assert [r['rank'] for r in result] == [1, 2]


def test_wrong_gallery_shape_and_nan_are_rejected():
    for x in [np.ones((15,256)), np.full((15,32,256),np.nan),np.zeros((15,32,256))]:
        with pytest.raises(ValueError): validate_tokens(x)
