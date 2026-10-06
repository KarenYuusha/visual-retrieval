import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from retrieval_common import load_subset, load_captions, middle_indices, clean_text, PREPROCESSING_V1, PREPROCESSING_V2
from evaluate_retrieval import compute_ranks, metrics_from_ranks


def test_combined_subset_ignores_seed_and_deduplicates(tmp_path):
    path = tmp_path / 'subset.json'
    path.write_text(json.dumps({'seed': 67, 'train': ['video3', 'video1'],
                                'test': ['video1', 'video2.mp4']}))
    assert load_subset(path) == ['video3', 'video1', 'video2']


def test_every_caption_is_kept_and_paired_with_video(tmp_path):
    path = tmp_path / 'MSRVTT_data.json'
    path.write_text(json.dumps({'sentences': [
        {'video_id': 'video1', 'caption': 'A man walks.', 'sen_id': 4},
        {'video_id': 'video2', 'caption': 'A dog runs.', 'sen_id': 5},
        {'video_id': 'video1', 'caption': 'A man walks.', 'sen_id': 6},
        {'video_id': 'video9', 'caption': 'Outside subset.', 'sen_id': 7},
    ]}))
    queries = load_captions(path, ['video1', 'video2'])
    assert [q['video_id'] for q in queries] == ['video1', 'video2', 'video1']
    assert [q['query_id'] for q in queries] == [4, 5, 6]
    assert queries[0]['text'] == 'A man walks.'


def test_missing_caption_fails_instead_of_changing_gallery(tmp_path):
    path = tmp_path / 'ann.json'
    path.write_text(json.dumps({'sentences': [
        {'video_id': 'video1', 'caption': 'A dog.'}]}))
    with pytest.raises(ValueError, match='video2'):
        load_captions(path, ['video1', 'video2'])


def test_uniform_segment_centers_and_short_video_padding():
    assert middle_indices(20, 4) == [2, 7, 12, 17]
    assert middle_indices(2, 4) == [0, 1, 1, 1]
    with pytest.raises(ValueError):
        middle_indices(0, 4)


def test_metrics_use_entire_gallery_and_full_mrr():
    videos = np.eye(12, dtype=np.float32)
    texts = np.zeros((4, 12), dtype=np.float32)
    desired = [1, 2, 6, 11]
    for i, rank in enumerate(desired):
        texts[i] = np.arange(12, 0, -1)
        texts[i, 0] = 12.5 - rank
    ranks = compute_ranks(texts, videos, [0, 0, 0, 0], batch_size=2)
    np.testing.assert_array_equal(ranks, desired)
    result = metrics_from_ranks(ranks)
    assert result['r1'] == 25
    assert result['r5'] == 50
    assert result['r10'] == 75
    assert result['mrr'] == pytest.approx((1 + 1/2 + 1/6 + 1/11) / 4)


def test_ties_are_broken_in_gallery_order():
    ranks = compute_ranks(np.ones((2, 3)), np.eye(3), [0, 2], batch_size=1)
    np.testing.assert_array_equal(ranks, [1, 3])


def test_preprocessing_matches_official_caption_cleaning():
    assert clean_text('A MAN, walks-down / road!') == 'a man walks down road'


def test_evaluation_cli_and_mapping(tmp_path):
    features = tmp_path / 'features'
    features.mkdir()
    np.savez(features / 'features.npz',
             video_features=np.eye(3, dtype=np.float32),
             text_features=np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32),
             video_ids=np.array(['video3', 'video1', 'video2']),
             text_video_ids=np.array(['video3', 'video2']))
    (features / 'manifest.json').write_text(json.dumps({
        'requested_video_ids': ['video3', 'video1', 'video2'],
        'failed_videos': [], 'protocol': 'combined_train_test_all_captions',
        'preprocessing': PREPROCESSING_V2}))
    (features / 'queries.json').write_text(json.dumps([
        {'video_id': 'video3', 'query_id': 1, 'text': 'First.'},
        {'video_id': 'video2', 'query_id': 2, 'text': 'Second.'}]))
    script = Path(__file__).parents[1] / 'evaluate_retrieval.py'
    subprocess.run([sys.executable, str(script), '--features-dir', str(features)], check=True)
    result = json.loads((features / 'evaluation.json').read_text())
    assert result['num_videos'] == 3
    assert result['num_queries'] == 2
    assert result['text_to_video']['r1'] == 100
    assert result['text_to_video']['mrr'] == 1
    assert result['gallery'] == 'train+test'


def test_incomplete_gallery_requires_explicit_opt_in(tmp_path):
    features = tmp_path / 'features'
    features.mkdir()
    np.savez(features / 'features.npz', video_features=np.array([[1, 0]], dtype=np.float32),
             text_features=np.array([[1, 0]], dtype=np.float32),
             video_ids=np.array(['video1']), text_video_ids=np.array(['video1']))
    (features / 'manifest.json').write_text(json.dumps({
        'requested_video_ids': ['video1', 'video2'], 'protocol': 'combined_train_test_all_captions',
        'preprocessing': PREPROCESSING_V2}))
    (features / 'queries.json').write_text(json.dumps([
        {'video_id': 'video1', 'query_id': 1, 'text': 'A person.'}]))
    script = Path(__file__).parents[1] / 'evaluate_retrieval.py'
    command = [sys.executable, str(script), '--features-dir', str(features)]
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert not (features / 'evaluation.json').exists()
    subprocess.run(command + ['--allow-incomplete'], check=True)
    report = json.loads((features / 'evaluation.json').read_text())
    assert report['complete'] is False
    assert report['omitted_video_ids'] == ['video2']


def test_evaluation_rejects_old_sep_bearing_text_embeddings(tmp_path):
    np.savez(tmp_path / 'features.npz', video_features=np.ones((1, 2), dtype=np.float32),
             text_features=np.ones((1, 2), dtype=np.float32),
             video_ids=np.array(['video1']), text_video_ids=np.array(['video1']))
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'requested_video_ids': ['video1'], 'protocol': 'combined_train_test_all_captions',
        'preprocessing': PREPROCESSING_V1}))
    (tmp_path / 'queries.json').write_text(json.dumps([
        {'video_id': 'video1', 'query_id': 1, 'text': 'A man.'}]))
    script = Path(__file__).parents[1] / 'evaluate_retrieval.py'
    result = subprocess.run([sys.executable, str(script), '--features-dir', str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'tokenizer' in result.stderr.lower()
    assert not (tmp_path / 'evaluation.json').exists()
