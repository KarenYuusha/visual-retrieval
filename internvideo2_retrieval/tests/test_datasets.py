import json
import numpy as np
import pytest
from dataset_adapters import load_dataset, segment_indices


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_vatex_combines_splits_and_keeps_all_english_captions(tmp_path):
    a, b = 'a_000001_000011', 'b_000002_000012'
    put(tmp_path, 'subset.json', {'seed': 67, 'train': [a], 'validation': [b, a], 'train_size': 1})
    put(tmp_path, 'raw_data/vatex_training_v1.0.json', [dict(videoID=a, enCap=['one', 'one'], chCap=['中文'])])
    put(tmp_path, 'raw_data/vatex_validation_v1.0.json', [dict(videoID=b, enCap=['two'])])
    data = load_dataset('vatex', tmp_path)
    assert [x['item_id'] for x in data.items] == [a, b]
    assert [x['text'] for x in data.queries] == ['one', 'one', 'two']
    assert data.splits == ['train', 'validation']
    assert len(data.annotation_paths) == 2


def activity(root):
    put(root, 'subset.json', {'train': ['v_abc'], 'validation': ['v_def']})
    put(root, 'raw_data/train.json', {'v_abc': dict(duration=10, timestamps=[[0, 2], [2, 10]], sentences=['first', 'second'])})
    put(root, 'raw_data/val_1.json', {'v_def': dict(duration=12, timestamps=[[1, 4]], sentences=['third'])})


def test_activitynet_distinct_segments_share_source(tmp_path):
    activity(tmp_path)
    data = load_dataset('activitynet_captions', tmp_path)
    assert [i['item_id'] for i in data.items] == ['v_abc__seg0000', 'v_abc__seg0001', 'v_def__seg0000']
    assert [q['video_id'] for q in data.queries] == [i['item_id'] for i in data.items]
    assert data.items[0]['path'] == data.items[1]['path']
    assert data.items[1]['start'] == 2
    assert data.items[1]['end'] == 10


def test_activitynet_video_mode_uses_one_paragraph(tmp_path):
    activity(tmp_path)
    data = load_dataset('activitynet_captions', tmp_path, activitynet_mode='video')
    assert [q['text'] for q in data.queries] == ['first second', 'third']
    assert len(data.items) == 2
    assert data.items[0]['start'] is None


def test_activitynet_rejects_misaligned_annotations(tmp_path):
    activity(tmp_path)
    put(tmp_path, 'raw_data/train.json', {'v_abc': dict(duration=10, timestamps=[[0, 2]], sentences=['first', 'second'])})
    with pytest.raises(ValueError, match='timestamps'):
        load_dataset('activitynet_captions', tmp_path)


def test_no_silent_missing_captions(tmp_path):
    put(tmp_path, 'subset.json', {'train': ['a'], 'test': []})
    put(tmp_path, 'vatex_training_v1.0.json', [dict(videoID='a', enCap=[])])
    with pytest.raises(ValueError, match='caption'):
        load_dataset('vatex', tmp_path)


def test_unsafe_subset_ids_rejected(tmp_path):
    put(tmp_path, 'subset.json', {'train': ['../outside'], 'test': []})
    with pytest.raises(ValueError, match='ID'):
        load_dataset('vatex', tmp_path)


def test_segments_sample_only_interval_and_reject_outside_video():
    assert segment_indices([i / 10 for i in range(100)], 2, 4) == [22, 27, 32, 37]
    assert segment_indices([i / 10 for i in range(100)], 9.9, 12) == [99, 99, 99, 99]
    with pytest.raises(ValueError):
        segment_indices([i / 10 for i in range(100)], 11, 12)
    with pytest.raises(ValueError):
        segment_indices([0, 0, 0], 1, 2)
    with pytest.raises(ValueError):
        segment_indices([0, 1, 2], np.nan, 2)


def test_decoder_respects_segment_time(tmp_path):
    import cv2
    from iv2_model import read_video_frames
    path = tmp_path / 'temporal.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (32, 24))
    assert writer.isOpened()
    for i in range(30):
        writer.write(np.full((24, 32, 3), i * 7, dtype=np.uint8))
    writer.release()
    rgb, indices, total = read_video_frames(path, start=1, end=2)
    assert total == 30
    assert indices == [10, 13, 15, 18]
    assert rgb.mean() > 60


def test_vfr_segment_sampling_uses_presentation_timestamps(monkeypatch):
    import cv2
    import iv2_model
    times = [0.0, .1, .2, .3, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    class Capture:
        def __init__(self, path):
            self.index = -1
        def isOpened(self):
            return True
        def grab(self):
            self.index += 1
            return self.index < len(times)
        def get(self, prop):
            return times[self.index] * 1000 if prop == cv2.CAP_PROP_POS_MSEC else 6.666666
        def retrieve(self):
            return True, np.full((2, 2, 3), self.index, dtype=np.uint8)
        def release(self):
            pass
    monkeypatch.setattr(iv2_model.cv2, 'VideoCapture', Capture)
    rgb, indices, count = iv2_model.read_video_frames('vfr.mp4', 0, 1)
    assert indices == [0, 1, 2, 3]
    assert all(times[i] < 1 for i in indices)
    assert count == len(times)
