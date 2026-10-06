import json
import numpy as np
import pytest
from visual_retrieval.evaluation.comparison import compare_bundles
from visual_retrieval.features.cache import atomic_npz
from visual_retrieval.common import write_json, PREPROCESSING_V2
from visual_retrieval.models.registry import MODEL_INFO


def bundle(root, model, changed=False):
    root.mkdir()
    atomic_npz(root/'features.npz', video_features=np.eye(2), text_features=np.eye(2),
               video_ids=np.array(['a','b']), text_video_ids=np.array(['a','b']))
    write_json(root/'queries.json', [dict(query_id='q1',video_id='a',text='changed' if changed else 'one'),
                                   dict(query_id='q2',video_id='b',text='two')])
    write_json(root/'manifest.json', dict(model_name=model, model_id=MODEL_INFO[model]['model_id'],
               preprocessing=MODEL_INFO[model]['preprocessing'], dataset='vatex',
               protocol='combined_subset_all_english_captions', requested_video_ids=['a','b'],
               complete=True, splits=['train','validation']))


def test_comparison_requires_identical_queries_and_full_galleries(tmp_path):
    a,b = tmp_path/'clip', tmp_path/'iv2'
    bundle(a,'clip'); bundle(b,'internvideo2')
    report = compare_bundles([a,b])
    assert report['results'][0]['text_to_video']['r1'] == 100
    queries=json.loads((b/'queries.json').read_text()); queries[0]['text']='changed'
    write_json(b/'queries.json',queries)
    with pytest.raises(ValueError, match='queries'):
        compare_bundles([a,b])


def test_comparison_rejects_missing_requested_items(tmp_path):
    a,b = tmp_path/'clip',tmp_path/'iv2'
    bundle(a,'clip');bundle(b,'internvideo2')
    manifest=json.loads((b/'manifest.json').read_text());manifest['requested_video_ids'].append('c')
    write_json(b/'manifest.json',manifest)
    with pytest.raises(ValueError,match='incomplete'):
        compare_bundles([a,b])


def test_comparison_rejects_changed_segment_intervals(tmp_path):
    a,b=tmp_path/'clip',tmp_path/'iv2'
    bundle(a,'clip');bundle(b,'internvideo2')
    for path,end in [(a,2),(b,3)]:
        manifest=json.loads((path/'manifest.json').read_text())
        manifest.update(dataset='activitynet',protocol='combined_subset_activitynet_segments',
                        gallery_items=[dict(item_id=v,source_video_id='source',start=0,end=end) for v in ['a','b']])
        write_json(path/'manifest.json',manifest)
    with pytest.raises(ValueError,match='gallery'):
        compare_bundles([a,b])
