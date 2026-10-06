import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from extract_features import atomic_npz, cached_video, ensure_cache_config
from retrieval_common import file_signature, sha256, PROTOCOL


def test_changed_video_invalidates_cache(tmp_path):
    source = tmp_path / 'video1.mp4'
    source.write_bytes(b'first')
    cache = tmp_path / 'video1.npz'
    atomic_npz(cache, feature=np.ones(512), source_signature=json.dumps(file_signature(source)))
    assert cached_video(cache, file_signature(source)).shape == (512,)
    source.write_bytes(b'changed source')
    assert cached_video(cache, file_signature(source)) is None


def test_truncated_cache_is_rebuilt(tmp_path):
    cache = tmp_path / 'broken.npz'
    cache.write_bytes(b'PK\x03\x04broken archive')
    assert cached_video(cache, {'size': 1}) is None


def test_orphan_caches_cannot_be_adopted_under_new_settings(tmp_path):
    cache = tmp_path / 'video_cache'
    cache.mkdir()
    atomic_npz(cache / 'video1.npz', feature=np.ones(512))
    with pytest.raises(ValueError, match='orphan'):
        ensure_cache_config(tmp_path, {'protocol': PROTOCOL})
    assert not (tmp_path / 'run_config.json').exists()


def test_tokenizer_fix_invalidates_only_text_and_combined_outputs(tmp_path):
    old_preprocessing = '4_middle_frames_uint8_bicubic224_imagenet_text_clean_max40_v1'
    new_preprocessing = '4_middle_frames_uint8_bicubic224_imagenet_text_clean_cls_only_max40_v2'
    old = {'protocol': PROTOCOL, 'checkpoint': 'same-weights', 'preprocessing': old_preprocessing}
    new = {**old, 'preprocessing': new_preprocessing}
    (tmp_path / 'run_config.json').write_text(json.dumps(old))
    cache = tmp_path / 'video_cache'
    cache.mkdir()
    atomic_npz(cache / 'video1.npz', feature=np.ones(512))
    atomic_npz(tmp_path / 'text_cache.npz', text_features=np.ones((1, 512)))
    atomic_npz(tmp_path / 'features.npz', video_features=np.ones((1, 512)))
    (tmp_path / 'evaluation.json').write_text('{"text_to_video":{"r1":4.805}}')
    ensure_cache_config(tmp_path, new)
    assert (cache / 'video1.npz').is_file()
    assert not (tmp_path / 'text_cache.npz').exists()
    assert not (tmp_path / 'features.npz').exists()
    assert not (tmp_path / 'evaluation.json').exists()
    assert (tmp_path / 'evaluation_before_tokenizer_fix.json').is_file()
    assert json.loads((tmp_path / 'run_config.json').read_text()) == new


def test_resume_packages_cached_features_and_evaluates_without_loading_model(tmp_path):
    root = tmp_path / 'data'
    videos = root / 'raw_videos'
    annotations = root / 'raw_data'
    output = tmp_path / 'output'
    cache = output / 'video_cache'
    videos.mkdir(parents=True)
    annotations.mkdir()
    cache.mkdir(parents=True)
    (root / 'subset.json').write_text(json.dumps({'seed': 67, 'train': ['video2'], 'test': ['video1']}))
    queries = [{'video_id': 'video1', 'caption': 'A dog runs.'},
               {'video_id': 'video2', 'caption': 'A man walks.'}]
    (annotations / 'MSRVTT_data.json').write_text(json.dumps({'sentences': queries}))
    feature1, feature2 = np.eye(512, dtype=np.float32)[:2]
    for video_id, feature in [('video1', feature1), ('video2', feature2)]:
        source = videos / f'{video_id}.mp4'
        source.write_bytes(b'fixture bytes; not decoded because cache exists')
        atomic_npz(cache / f'{video_id}.npz', feature=feature,
                   source_signature=json.dumps(file_signature(source)))
    atomic_npz(output / 'text_cache.npz', text_features=np.stack([feature1, feature2]),
               text_video_ids=np.array(['video1', 'video2']))
    # If the model is loaded by mistake, this invalid checkpoint would fail.
    checkpoint = tmp_path / 'dummy.pt'
    checkpoint.write_bytes(b'not a checkpoint')
    from iv2_model import MODEL_ID, SOURCE_COMMIT
    # A legitimate resumed run has the settings record written before any cache.
    (output / 'run_config.json').write_text(json.dumps({
        'model_id': MODEL_ID, 'source_commit': SOURCE_COMMIT, 'protocol': PROTOCOL,
        'subset_sha256': sha256(root / 'subset.json'),
        'annotations_sha256': sha256(annotations / 'MSRVTT_data.json'),
        'checkpoint': file_signature(checkpoint), 'tokenizer': 'google-bert/bert-large-uncased',
        'precision': 'auto', 'device': 'cpu',
        'preprocessing': '4_middle_frames_uint8_bicubic224_imagenet_text_clean_cls_only_max40_v2'}))
    scripts = Path(__file__).parents[1]
    command = [sys.executable, str(scripts / 'extract_features.py'), '--data-root', str(root),
               '--output-dir', str(output), '--checkpoint', str(checkpoint), '--device', 'cpu']
    subprocess.run(command, check=True)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['reused_video_embeddings'] == 2
    assert manifest['successful_video_ids'] == ['video2', 'video1']
    subprocess.run([sys.executable, str(scripts / 'evaluate_retrieval.py'),
                    '--features-dir', str(output)], check=True)
    report = json.loads((output / 'evaluation.json').read_text())
    assert report['text_to_video']['r1'] == 100
    assert report['text_to_video']['mrr'] == 1


def test_data_check_does_not_need_a_checkpoint(tmp_path):
    (tmp_path / 'raw_videos').mkdir()
    (tmp_path / 'raw_videos' / 'video1.mp4').write_bytes(b'not decoded during preflight')
    (tmp_path / 'subset.json').write_text(json.dumps({'seed': 67, 'train': ['video1'], 'test': []}))
    (tmp_path / 'MSRVTT_data.json').write_text(json.dumps({'sentences': [
        {'video_id': 'video1', 'caption': 'A person.'}]}))
    script = Path(__file__).parents[1] / 'extract_features.py'
    result = subprocess.run([sys.executable, str(script), '--data-root', str(tmp_path),
                             '--check-data'], check=True, capture_output=True, text=True)
    assert '1 unique videos; 1 caption queries' in result.stdout
    assert not (tmp_path / 'features').exists()


@pytest.mark.parametrize('dataset,mode', [('vatex', 'segments'), ('activitynet_captions', 'segments'), ('activitynet_captions', 'video')])
def test_all_dataset_cached_cli_pipeline(tmp_path, dataset, mode):
    from dataset_adapters import load_dataset
    from iv2_model import MODEL_ID, SOURCE_COMMIT
    from retrieval_common import PREPROCESSING_V2
    root = tmp_path / 'data'
    root.mkdir()
    (root / 'raw_videos').mkdir()
    if dataset == 'vatex':
        ids = ['a_000000_000010', 'b_000000_000010']
        annotations = [{'videoID': v, 'enCap': ['one', 'two']} for v in ids]
        (root / 'vatex_training_v1.0.json').write_text(json.dumps(annotations))
    else:
        ids = ['v_abc', 'v_def']
        row = dict(duration=3, timestamps=[[0, 1], [1, 3]], sentences=['one', 'two'])
        (root / 'train.json').write_text(json.dumps({v: row for v in ids}))
    (root / 'subset.json').write_text(json.dumps(dict(train=ids[:1], validation=ids[1:])))
    for v in ids:
        (root / 'raw_videos' / f'{v}.mp4').write_bytes(b'fixture')
    data = load_dataset(dataset, root, activitynet_mode=mode)
    output = tmp_path / 'out'
    cache = output / 'video_cache'
    cache.mkdir(parents=True)
    features = np.eye(512, dtype=np.float32)[:len(data.items)]
    for item, feature in zip(data.items, features):
        signature = file_signature(item['path'])
        if item['start'] is not None:
            signature.update(start=item['start'], end=item['end'])
        atomic_npz(cache / f"{item['item_id']}.npz", feature=feature, source_signature=json.dumps(signature))
    lookup = {item['item_id']: f for item, f in zip(data.items, features)}
    labels = [q['video_id'] for q in data.queries]
    atomic_npz(output / 'text_cache.npz', text_features=np.stack([lookup[v] for v in labels]), text_video_ids=np.array(labels))
    checkpoint = tmp_path / 'dummy.pt'
    checkpoint.write_bytes(b'no model should load')
    config = dict(model_id=MODEL_ID, source_commit=SOURCE_COMMIT, protocol=data.protocol,
                  subset_sha256=sha256(root / 'subset.json'), annotations_sha256=sha256(data.annotation_paths[0]),
                  checkpoint=file_signature(checkpoint), tokenizer='google-bert/bert-large-uncased',
                  precision='auto', device='cpu', preprocessing=PREPROCESSING_V2,
                  dataset=dataset, activitynet_mode=mode if dataset == 'activitynet_captions' else None)
    (output / 'run_config.json').write_text(json.dumps(config))
    scripts = Path(__file__).parents[1]
    subprocess.run([sys.executable, str(scripts / 'extract_features.py'), '--dataset', dataset,
                    '--activitynet-mode', mode, '--data-root', str(root), '--output-dir', str(output),
                    '--device', 'cpu', '--checkpoint', str(checkpoint)], check=True)
    subprocess.run([sys.executable, str(scripts / 'evaluate_retrieval.py'), '--features-dir', str(output)], check=True)
    report = json.loads((output / 'evaluation.json').read_text())
    assert report['dataset'] == dataset
    assert report['num_source_videos'] == 2
    assert report['num_videos'] == len(data.items)
    assert report['text_to_video']['r1'] == 100
    assert report['text_to_video']['mrr'] == 1
