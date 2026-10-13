import json
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from visual_retrieval.common import write_json
from visual_retrieval.models.registry import MODEL_INFO


def saved_gallery(root, model='clip', ids=('same',), features=None, checkpoint='trained', precision='auto'):
    folder = root / 'features' / model
    folder.mkdir(parents=True)
    (root / 'raw_videos').mkdir(exist_ok=True)
    items = []
    for item in ids:
        source = root / 'raw_videos' / f'{item}.mp4'
        source.write_bytes(b'media fixture')
        items.append(dict(item_id=item, source_video_id=item, path=str(source), start=None, end=None))
    np.savez(folder / 'features.npz', video_ids=np.array(ids), video_features=np.array(features if features is not None else [[1., 0.]], dtype=np.float32))
    info = MODEL_INFO[model]
    config = dict(model_name=model, model_id=info['model_id'], preprocessing=info['preprocessing'],
                  source_commit=info['source_commit'], precision=precision, device='cuda',
                  tokenizer='matching-tokenizer', checkpoint=dict(path=f'/weights/{checkpoint}', size=123, mtime_ns=1))
    write_json(folder / 'run_config.json', config)
    write_json(folder / 'manifest.json', dict(**config, protocol='combined_train_test_all_captions',
               effective_precision='fp32', gallery_items=items, requested_video_ids=list(ids) + ['omitted'], successful_video_ids=list(ids), complete=False))
    return folder


def make_spec(root, key, models=('clip',)):
    from visual_retrieval.demo.config import DatasetSpec
    return DatasetSpec(key, key.upper(), root, {m: root / 'features' / m for m in models})


class Encoder:
    def encode_text(self, texts):
        return np.array([[0., 1.] if 'beta' in text else [1., 0.] for text in texts], dtype=np.float32)


def test_combined_search_namespaces_ids_and_matches_cosine_order(tmp_path):
    from visual_retrieval.demo.service import SearchService
    a, b = tmp_path / 'a', tmp_path / 'b'
    saved_gallery(a, features=[[1., 0.]])
    saved_gallery(b, features=[[.8, .6]])
    service = SearchService([make_spec(a, 'a'), make_spec(b, 'b')], device='cpu', encoder_loader=lambda *a: Encoder())
    results = service.search('alpha', 'clip', top_k=10)
    assert [(r['dataset'], r['item_id']) for r in results] == [('a', 'same'), ('b', 'same')]
    assert results[0]['key'] != results[1]['key']
    assert results[0]['cosine_similarity'] == pytest.approx(1)
    assert results[1]['cosine_similarity'] == pytest.approx(.8)
    assert [r['rank'] for r in service.search('alpha', 'clip', dataset='b')] == [1]
    assert service.gallery_counts('clip') == {'a': 1, 'b': 1}
    assert service.omitted_counts('clip') == {'a': 1, 'b': 1}


@pytest.mark.parametrize('change', ['checkpoint', 'precision', 'tokenizer', 'dimension'])
def test_combined_search_rejects_incompatible_spaces_but_single_gallery_works(tmp_path, change):
    from visual_retrieval.demo.service import SearchService
    a, b = tmp_path / 'a', tmp_path / 'b'
    saved_gallery(a)
    folder = saved_gallery(b, checkpoint='different' if change == 'checkpoint' else 'trained',
                           precision='fp16' if change == 'precision' else 'auto',
                           features=[[1,0,0]] if change == 'dimension' else None)
    if change == 'tokenizer':
        for filename in ('run_config.json', 'manifest.json'):
            cfg = json.loads((folder / filename).read_text()); cfg['tokenizer'] = 'different'
            write_json(folder / filename, cfg)
    service = SearchService([make_spec(a, 'a'), make_spec(b, 'b')], device='cpu', encoder_loader=lambda *a: Encoder())
    with pytest.raises(ValueError, match='compatible'):
        service.search('alpha', 'clip')
    assert service.search('alpha', 'clip', 'a')[0]['dataset'] == 'a'


def test_one_active_encoder_reused_and_released_before_model_switch(tmp_path):
    from visual_retrieval.demo.service import SearchService
    saved_gallery(tmp_path / 'a', 'clip'); saved_gallery(tmp_path / 'a', 'internvideo2')
    refs, loads = [], []
    def loader(folder, device):
        assert all(ref() is None for ref in refs), 'Old encoder must be released before loading another'
        encoder = Encoder(); refs.append(weakref.ref(encoder)); loads.append(folder)
        return encoder
    service = SearchService([make_spec(tmp_path / 'a', 'a', ('clip', 'internvideo2'))], device='cpu', encoder_loader=loader)
    service.search('alpha', 'clip'); service.search('beta', 'clip')
    assert len(loads) == 1
    service.search('alpha', 'internvideo2')
    assert len(loads) == 2
    service.close()
    assert all(ref() is None for ref in refs)


def test_parallel_searches_are_serialized_and_keep_model_identity(tmp_path):
    from visual_retrieval.demo.service import SearchService
    saved_gallery(tmp_path / 'a', 'clip'); saved_gallery(tmp_path / 'a', 'internvideo2')
    refs = []
    def loader(folder, device):
        assert all(ref() is None for ref in refs)
        encoder = Encoder(); refs.append(weakref.ref(encoder)); return encoder
    service = SearchService([make_spec(tmp_path / 'a', 'a', ('clip', 'internvideo2'))], device='cpu', encoder_loader=loader)
    with ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(service.search, 'alpha', model) for model in ('clip', 'internvideo2')]
        assert all(job.result()[0]['item_id'] == 'same' for job in jobs)


def test_missing_gallery_excluded_and_invalid_media_metadata_rejected(tmp_path):
    from visual_retrieval.demo.service import SearchService
    good, bad = tmp_path / 'good', tmp_path / 'bad'
    saved_gallery(good)
    folder = saved_gallery(bad)
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest['gallery_items'][0].update(start=2, end=1)
    write_json(folder / 'manifest.json', manifest)
    service = SearchService([make_spec(good, 'good', ('clip','clip4clip')), make_spec(bad,'bad')], device='cpu', encoder_loader=lambda *a: Encoder())
    assert service.available_models == ['clip']
    assert service.dataset_choices('clip') == [('All datasets', 'all'), ('GOOD', 'good')]
    assert len(service.issues) == 2
    with pytest.raises(ValueError, match='description'):
        service.search('   ', 'clip')


def test_config_defaults_and_custom_dataset_relative_paths(tmp_path):
    from visual_retrieval.demo.config import DemoConfig
    default = DemoConfig.defaults(tmp_path)
    assert [x.key for x in default.datasets] == ['msrvtt','vatex','activitynet']
    assert default.datasets[0].root == tmp_path / 'msr_vtt'
    path = tmp_path / 'demo.yaml'
    path.write_text('default_model: clip\ndatasets:\n  extra:\n    label: More videos\n    root: extra_data\n    features:\n      clip: saved_clip\n')
    config = DemoConfig.from_yaml(path)
    assert config.default_model == 'clip'
    assert config.datasets[0].feature_dirs['clip'] == tmp_path / 'extra_data' / 'saved_clip'
    assert config.datasets[0].key == 'extra'


def test_no_gallery_and_path_traversal_are_clear_errors(tmp_path):
    from visual_retrieval.demo.config import DemoConfig
    from visual_retrieval.demo.service import SearchService
    with pytest.raises(ValueError, match='No searchable galleries'):
        SearchService([make_spec(tmp_path / 'empty', 'empty')], device='cpu')
    path = tmp_path / 'invalid.yaml'
    path.write_text('datasets:\n  all:\n    root: data\n')
    with pytest.raises(ValueError, match='dataset key'):
        DemoConfig.from_yaml(path)
    folder = saved_gallery(tmp_path / 'unsafe')
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest['gallery_items'][0]['path'] = '../private.txt'
    write_json(folder / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='No searchable galleries'):
        SearchService([make_spec(tmp_path / 'unsafe', 'unsafe')], device='cpu')

@pytest.mark.parametrize('bounds', [(None, None), (True, 2)])
def test_segment_gallery_requires_numeric_bounds(tmp_path, bounds):
    from visual_retrieval.demo.service import SearchService
    folder = saved_gallery(tmp_path / 'a')
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest['protocol'] = 'combined_subset_activitynet_segments'
    manifest['gallery_items'][0].update(start=bounds[0], end=bounds[1])
    write_json(folder / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='No searchable'):
        SearchService([make_spec(tmp_path / 'a', 'a')])


def test_combined_search_checks_effective_precision(tmp_path):
    from visual_retrieval.demo.service import SearchService
    for key, precision in [('a', 'bf16'), ('b', 'fp16')]:
        folder = saved_gallery(tmp_path / key)
        manifest = json.loads((folder / 'manifest.json').read_text())
        manifest['effective_precision'] = precision
        write_json(folder / 'manifest.json', manifest)
    service = SearchService([make_spec(tmp_path / 'a', 'a'), make_spec(tmp_path / 'b', 'b')], 'cpu', lambda *a: Encoder())
    with pytest.raises(ValueError, match='compatible'):
        service.search('alpha', 'clip')


def test_unknown_precision_prevents_combined_search(tmp_path):
    from visual_retrieval.demo.service import SearchService
    for key in ('a', 'b'):
        folder = saved_gallery(tmp_path / key)
        manifest = json.loads((folder / 'manifest.json').read_text())
        manifest.pop('effective_precision')
        write_json(folder / 'manifest.json', manifest)
    service = SearchService([make_spec(tmp_path / 'a', 'a'), make_spec(tmp_path / 'b', 'b')], 'cpu', lambda *a: Encoder())
    assert service.search('alpha', 'clip', 'a')
    with pytest.raises(ValueError, match='compatible'):
        service.search('alpha', 'clip')


def test_cuda_query_precision_must_match_gallery(tmp_path):
    from visual_retrieval.demo.service import SearchService
    saved_gallery(tmp_path / 'a')
    encoder = Encoder(); encoder.effective_precision = 'fp16'
    service = SearchService([make_spec(tmp_path / 'a', 'a')], 'cuda', lambda *a: encoder)
    with pytest.raises(ValueError, match='Query precision'):
        service.search('alpha', 'clip')
    assert service._encoder is None
