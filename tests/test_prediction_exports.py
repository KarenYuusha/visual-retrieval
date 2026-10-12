import csv
import json

import numpy as np
import pytest

from visual_retrieval.common import write_json
from visual_retrieval.evaluation.baseline_evaluation import evaluate_directory
from visual_retrieval.features.cache import atomic_npz
from visual_retrieval.models.registry import MODEL_INFO


def make_bundle(root):
    atomic_npz(root / 'features.npz', video_features=np.eye(3, dtype=np.float32),
               text_features=np.array([[1,0,0], [0,1,0], [1,1,0], [0,0,1]], dtype=np.float32),
               video_ids=np.array(['a', 'b', 'c']), text_video_ids=np.array(['a', 'a', 'b', 'c']))
    write_json(root / 'queries.json', [dict(query_id=i, video_id=v, text=f'Caption {i}: đi bộ\nnext line')
               for i, v in enumerate(['a', 'a', 'b', 'c'])])
    info = MODEL_INFO['clip']
    write_json(root / 'manifest.json', dict(model_name='clip', model_id=info['model_id'],
               preprocessing=info['preprocessing'], dataset='activitynet_captions',
               protocol='combined_subset_activitynet_segments', requested_video_ids=['a', 'b', 'c'],
               splits=['train', 'validation'], gallery_items=[
                   dict(item_id='a', source_video_id='same_source', path='S:/videos/same.mp4', start=0, end=1),
                   dict(item_id='b', source_video_id='same_source', path='S:/videos/same.mp4', start=1, end=2),
                   dict(item_id='c', source_video_id='other', path='S:/videos/other.mp4', start=0, end=2)]))


def test_evaluation_exports_ground_truth_predictions_scores_and_stable_candidates(tmp_path):
    make_bundle(tmp_path)
    report = evaluate_directory(tmp_path, batch_size=2)
    with (tmp_path / 'query_ranks.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert 'predicted_video_id' in rows[0], 'Evaluator must save predicted labels'
    assert [row['ground_truth_video_id'] for row in rows] == ['a', 'a', 'b', 'c']
    assert [row['predicted_video_id'] for row in rows] == ['a', 'b', 'a', 'c']
    assert [row['is_correct'] for row in rows] == ['True', 'False', 'False', 'True']
    assert rows[1]['ground_truth_start'] == '0' and rows[1]['predicted_start'] == '1'
    assert rows[1]['ground_truth_path'] == rows[1]['predicted_path'] == 'S:/videos/same.mp4'
    assert rows[1]['caption'] == 'Caption 1: đi bộ\nnext line'
    assert float(rows[1]['ground_truth_score']) == 0
    assert float(rows[1]['predicted_score']) == 1
    assert float(rows[1]['score_gap']) == 1
    records = [json.loads(line) for line in (tmp_path / 'predictions.jsonl').read_text().splitlines()]
    assert len(records) == 4
    assert records[2]['ground_truth_video_id'] == 'b' and records[2]['is_correct'] is False
    assert records[2]['rank'] == 2 and records[2]['score_gap'] == 0
    assert [candidate['video_id'] for candidate in records[2]['top_k']] == ['a', 'b', 'c']
    assert records[1]['ground_truth_item']['start'] == 0
    assert records[1]['predicted_item']['start'] == 1
    assert report['text_to_video'] == dict(r1=50., r5=100., r10=100., mrr=.75)
    assert report['prediction_export']['top_k'] == 3


def test_top_k_export_does_not_truncate_ground_truth_rank(tmp_path):
    make_bundle(tmp_path)
    evaluate_directory(tmp_path, top_k=1)
    records = [json.loads(line) for line in (tmp_path / 'predictions.jsonl').read_text().splitlines()]
    assert len(records[1]['top_k']) == 1
    assert records[1]['rank'] == 2 and records[1]['ground_truth_video_id'] == 'a'


def test_partial_evaluation_exports_only_available_queries_and_reports_omission(tmp_path):
    make_bundle(tmp_path)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest['requested_video_ids'].append('missing')
    write_json(tmp_path / 'manifest.json', manifest)
    report = evaluate_directory(tmp_path, allow_incomplete=True)
    assert report['omitted_video_ids'] == ['missing']
    assert (tmp_path / 'predictions.jsonl').is_file()
    assert len((tmp_path / 'predictions.jsonl').read_text().splitlines()) == 4


def test_legacy_bundle_exports_labels_without_inventing_video_paths(tmp_path):
    make_bundle(tmp_path)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    del manifest['gallery_items']
    write_json(tmp_path / 'manifest.json', manifest)
    evaluate_directory(tmp_path)
    record = json.loads((tmp_path / 'predictions.jsonl').read_text().splitlines()[0])
    assert record['ground_truth_video_id'] == 'a'
    assert record['ground_truth_item'] is None and record['predicted_item'] is None


def test_top_k_must_be_positive(tmp_path):
    make_bundle(tmp_path)
    with pytest.raises(ValueError, match='top_k'):
        evaluate_directory(tmp_path, top_k=0)


def test_numpy_integer_top_k_can_be_saved_to_json(tmp_path):
    make_bundle(tmp_path)
    report = evaluate_directory(tmp_path, top_k=np.int64(2))
    assert report['prediction_export']['requested_top_k'] == 2
    assert json.loads((tmp_path / 'evaluation.json').read_text())['prediction_export']['top_k'] == 2
