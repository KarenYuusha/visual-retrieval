import threading
import json
import sys

import cv2
import numpy as np

from visual_retrieval.data import video_sampling


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (32, 24))
    assert writer.isOpened()
    for i in range(30):
        writer.write(np.full((24, 32, 3), i * 7, dtype=np.uint8))
    writer.release()


def test_grouped_sampling_matches_individual_segments_with_only_two_opens(tmp_path, monkeypatch):
    path = tmp_path / 'video.mp4'
    make_video(path)
    intervals = [(0, 1), (.7, 2.2), (2.8, 3.0), (5, 6), (None, None)]
    expected = [video_sampling.read_video_frames(path, a, b, 12) for a, b in intervals[:3]]
    expected.append(None)
    expected.append(video_sampling.read_video_frames(path, count_frames=12))
    original = cv2.VideoCapture
    opened = []
    def capture(*args):
        opened.append(args)
        return original(*args)
    monkeypatch.setattr(cv2, 'VideoCapture', capture)
    assert hasattr(video_sampling, 'read_video_segments'), 'Missing shared source decoder'
    results = video_sampling.read_video_segments(path, intervals, 12)
    assert len(opened) == 2
    # Independent index oracle protects the center/boundary/padding protocol.
    assert results[0][1] == list(range(10)) + [9, 9]
    assert results[1][1] == [7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20]
    assert results[2][1] == [28] + [29] * 11
    for result, reference in zip(results, expected):
        if reference is None:
            assert isinstance(result, ValueError)
        else:
            rgb, indices, count = result
            np.testing.assert_array_equal(rgb, reference[0])
            assert indices == reference[1] and count == reference[2]


def test_prefetch_prepares_next_source_while_consumer_is_busy():
    from visual_retrieval.features import pipeline
    prepared = threading.Event()
    def decode(value):
        if value == 1:
            prepared.set()
        return value
    assert hasattr(pipeline, 'prefetch_sources'), 'Missing bounded decoder prefetch'
    stream = pipeline.prefetch_sources([0, 1, 2], decode, workers=2)
    try:
        assert next(stream) == 0
        assert prepared.wait(2), 'Next source must decode before the consumer asks for it'
        assert list(stream) == [1, 2]
    finally:
        stream.close()


def test_one_bad_requested_frame_does_not_discard_other_segments(tmp_path, monkeypatch):
    path = tmp_path / 'damaged.mp4'
    make_video(path)
    original = cv2.VideoCapture
    class Capture:
        def __init__(self, path):
            self.cap = original(path)
        def __getattr__(self, name):
            return getattr(self.cap, name)
        def retrieve(self):
            if int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) == 16:
                return False, None
            return self.cap.retrieve()
    monkeypatch.setattr(cv2, 'VideoCapture', Capture)
    results = video_sampling.read_video_segments(path, [(0, 1), (1, 2), (2, 3)], 12)
    assert not isinstance(results[0], Exception)
    assert isinstance(results[1], ValueError)
    assert not isinstance(results[2], Exception)


def test_extraction_batches_segments_and_resumes_without_decoding(tmp_path, monkeypatch):
    from visual_retrieval.features import extraction
    root = tmp_path / 'dataset'
    (root / 'raw_videos').mkdir(parents=True)
    make_video(root / 'raw_videos' / 'v_one.mp4')
    (root / 'subset.json').write_text(json.dumps({'train': ['v_one'], 'validation': []}))
    (root / 'train.json').write_text(json.dumps({'v_one': dict(duration=3, timestamps=[[0, 1], [1, 2], [2, 3]], sentences=['a', 'b', 'c'])}))
    checkpoint = tmp_path / 'weights'
    checkpoint.mkdir()
    (checkpoint / 'pytorch_model.bin').write_bytes(b'fixture')
    output = tmp_path / 'output'
    batches = []
    class Encoder:
        effective_precision = 'fp32'
        def encode_videos(self, clips):
            batches.append(len(clips))
            return np.eye(512, dtype=np.float32)[:len(clips)]
        def encode_text(self, texts):
            return np.eye(512, dtype=np.float32)[:len(texts)]
    monkeypatch.setattr(extraction, 'create_encoder', lambda *args: Encoder())
    monkeypatch.setattr(sys, 'argv', ['extract', '--model', 'clip', '--dataset', 'activitynet', '--data-root', str(root),
                       '--checkpoint', str(checkpoint), '--device', 'cpu', '--output-dir', str(output),
                       '--video-batch-size', '2', '--decode-workers', '2'])
    extraction.main()
    assert batches == [2, 1]
    with np.load(output / 'features.npz') as bundle:
        assert list(bundle['video_ids']) == ['v_one__seg0000', 'v_one__seg0001', 'v_one__seg0002']
    batches.clear()
    (output / 'video_cache' / 'v_one__seg0001.npz').unlink()
    extraction.main()
    assert batches == [1]
    with np.load(output / 'features.npz') as bundle:
        assert list(bundle['video_ids']) == ['v_one__seg0000', 'v_one__seg0001', 'v_one__seg0002']
    assert json.loads((output / 'manifest.json').read_text())['reused_video_embeddings'] == 2
    batches.clear()
    def forbidden(*args):
        raise AssertionError('Cached run must not load model or decode videos')
    monkeypatch.setattr(extraction, 'create_encoder', forbidden)
    monkeypatch.setattr(cv2, 'VideoCapture', forbidden)
    extraction.main()
    assert not batches
    assert json.loads((output / 'manifest.json').read_text())['reused_video_embeddings'] == 3


def test_synchronous_prefetch_preserves_results():
    from visual_retrieval.features.pipeline import prefetch_sources
    assert list(prefetch_sources([1, 2, 3], lambda x: x * 2, workers=0)) == [2, 4, 6]


def test_prefetch_does_not_queue_the_entire_dataset():
    from visual_retrieval.features.pipeline import prefetch_sources
    consumed = []
    def sources():
        for value in range(100):
            consumed.append(value)
            yield value
    stream = prefetch_sources(sources(), lambda x: x, workers=2)
    try:
        assert next(stream) == 0
        assert consumed == [0, 1, 2]  # Current result plus two future sources.
    finally:
        stream.close()


def test_runner_forwards_decoding_and_video_batch_options():
    from visual_retrieval.cli.run_baselines import make_parser, commands_for
    args = make_parser().parse_args(['--decode-workers', '4', '--video-batch-size', '8'])
    command = commands_for(args, 'activitynet', 'clip')['extract']
    assert command[command.index('--decode-workers') + 1] == '4'
    assert command[command.index('--video-batch-size') + 1] == '8'
