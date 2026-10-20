import json
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from test_webvid_covr_evaluation import make_subset


def test_video_batches_preserve_order_and_resume_across_tuning_changes(tmp_path, monkeypatch):
    from visual_retrieval.data.webvid_covr import load_subset
    from visual_retrieval.features import webvid_covr as extraction
    make_subset(tmp_path)
    data = load_subset(tmp_path)
    weights = tmp_path/'weights'; weights.write_bytes(b'weights')
    decoded, batches = [], []
    ready = threading.Event()
    main = threading.get_ident()
    def decode(path, count_frames):
        assert threading.get_ident() != main
        id = Path(path).stem
        decoded.append(id)
        if id == '201': ready.set()
        return np.full((count_frames, 2, 2, 3), int(id) % 100, np.uint8), list(range(count_frames)), count_frames
    class Encoder:
        effective_precision = 'fp32'
        def encode_videos(self, clips):
            assert threading.get_ident() == main
            assert ready.wait(2), 'Decoding must run ahead of GPU inference'
            batches.append(len(clips))
            result = np.ones((len(clips), 512), np.float32)
            for row, clip in zip(result, clips): row[0] = int(clip[0, 0, 0, 0]) + 1
            return result
        def encode_text(self, texts): return np.ones((len(texts), 512), np.float32)
    monkeypatch.setattr(extraction, 'read_video_frames', decode)
    kwargs = dict(device='cpu', factory=lambda *a: Encoder(), resolver=lambda *a: (weights, 'tokenizer'))
    out = tmp_path/'out'
    extraction.extract_baseline(data, 'clip', out, video_batch_size=2, decode_workers=2, **kwargs)
    assert batches == [2, 1]
    with np.load(out/'features.npz') as archive:
        assert archive['video_ids'].tolist() == ['100', '200', '201']
        expected = np.ones((3, 512), np.float32); expected[:, 0] = [1, 1, 2]
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(archive['video_features'], expected, atol=1e-7)
    decoded.clear(); batches.clear()
    extraction.extract_baseline(data, 'clip', out, video_batch_size=4, decode_workers=0, **kwargs)
    assert not decoded and not batches


def test_covr2_frame_batches_match_single_frame_with_partial_batch():
    from visual_retrieval.models.covr2_encoder import CoVR2Encoder
    from visual_retrieval.models.covr2_vendor.Qformer import BertConfig, BertLMHeadModel
    torch.manual_seed(17)
    cfg = BertConfig(vocab_size=101, hidden_size=24, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=48, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    cfg.encoder_width, cfg.add_cross_attention, cfg.cross_attention_freq, cfg.query_length = 16, True, 1, 2
    encoder = CoVR2Encoder.__new__(CoVR2Encoder)
    encoder.gallery_mode = True; encoder.device = torch.device('cpu')
    encoder.qformer = BertLMHeadModel(cfg).eval()
    encoder.queries = torch.randn(1, 2, 24)
    encoder.vision_proj = torch.nn.Linear(24, 8).eval()
    batches = []
    def image(frames):
        batches.append(len(frames))
        return torch.stack(frames)
    encoder._image = image
    frames = list(torch.randn(5, 4, 16))
    encoder.frame_batch_size = 1
    expected = encoder.encode_frames(frames)
    batches.clear(); encoder.frame_batch_size = 2
    actual = encoder.encode_frames(frames)
    assert batches == [2, 2, 1]
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize('broken_model', [False, True])
def test_failed_decode_keeps_other_ids_aligned_and_model_errors_stop_workers(tmp_path, monkeypatch, broken_model):
    from visual_retrieval.data.webvid_covr import load_subset
    from visual_retrieval.features import webvid_covr as extraction
    make_subset(tmp_path); data = load_subset(tmp_path)
    weights = tmp_path/'weights'; weights.write_bytes(b'weights')
    workers = set()
    def decode(path, count_frames):
        workers.add(threading.current_thread())
        if Path(path).stem == '200': raise ValueError('damaged video')
        return np.zeros((count_frames, 2, 2, 3), np.uint8), list(range(count_frames)), count_frames
    class Encoder:
        effective_precision = 'fp32'
        def encode_videos(self, clips):
            if broken_model: raise RuntimeError('model failed')
            return np.ones((len(clips), 512), np.float32)
        def encode_text(self, texts): return np.ones((len(texts), 512), np.float32)
    monkeypatch.setattr(extraction, 'read_video_frames', decode)
    out = tmp_path/'out'
    kwargs = dict(device='cpu', factory=lambda *a: Encoder(), resolver=lambda *a: (weights, 'tokenizer'),
                  video_batch_size=2, decode_workers=2, allow_incomplete=True)
    if broken_model:
        with pytest.raises(RuntimeError, match='model failed'):
            extraction.extract_baseline(data, 'clip', out, **kwargs)
        assert not (out/'features.npz').exists()
    else:
        manifest = extraction.extract_baseline(data, 'clip', out, **kwargs)
        assert manifest['successful_video_ids'] == ['100', '201']
        assert manifest['failed_videos'] == [dict(video_id='200', error='damaged video')]
        with np.load(out/'features.npz') as archive:
            assert archive['video_ids'].tolist() == ['100', '201']
            assert archive['text_video_ids'].tolist() == ['201']
    assert workers and all(not worker.is_alive() for worker in workers)


def test_covr2_gallery_prefetches_and_keeps_cache_identity(tmp_path, monkeypatch):
    from visual_retrieval.composed import prepare as gallery
    from visual_retrieval.data.webvid_covr import load_subset
    make_subset(tmp_path); data = load_subset(tmp_path)
    source = tmp_path/'source.json'
    source.write_text(json.dumps(dict(gallery_items=[dict(i, path=str(i['path'])) for i in data.items])))
    weights = tmp_path/'weights'; weights.write_bytes(b'weights')
    ready = threading.Event(); calls = []
    def decode(path, intervals, count):
        assert threading.current_thread() is not threading.main_thread()
        if Path(path).stem == '200': ready.set()
        return [(np.ones((15, 2, 2, 3), np.uint8), list(range(15)), 15)]
    class Encoder:
        def __init__(self, *a, **kw): calls.append(kw['frame_batch_size'])
        def encode_frames(self, frames):
            assert ready.wait(2)
            tokens = np.zeros((15, 32, 256), np.float32); tokens[:, :, 0] = 1
            return tokens
    monkeypatch.setattr(gallery, 'read_video_segments', decode)
    out = tmp_path/'gallery'
    first = gallery.prepare(tmp_path, source, weights, out, 'cpu', encoder_factory=Encoder,
                            decode_workers=2, frame_batch_size=2)
    def forbidden(*a, **kw): raise AssertionError('Resume must skip models and decoding')
    monkeypatch.setattr(gallery, 'read_video_segments', forbidden)
    second = gallery.prepare(tmp_path, source, weights, out, 'cpu', encoder_factory=forbidden,
                             decode_workers=0, frame_batch_size=4)
    assert first['identity'] == second['identity'] and calls == [2]


def test_pipeline_forwards_throughput_controls(tmp_path, monkeypatch):
    from visual_retrieval.cli import webvid_covr as cli
    make_subset(tmp_path)
    weights = tmp_path/'weights'; weights.write_bytes(b'weights')
    baseline, gallery = [], []
    monkeypatch.setattr(cli, 'extract_baseline', lambda *a, **kw: baseline.append(kw))
    monkeypatch.setattr(cli, 'prepare_gallery', lambda *a, **kw: (gallery.append(kw) or {}))
    monkeypatch.setattr(cli, 'extract_queries', lambda *a, **kw: None)
    assert cli.main(['--root', str(tmp_path), '--stage', 'extract', '--device', 'cpu',
        '--clip4clip-checkpoint', str(weights), '--covr2-base-checkpoint', str(weights),
        '--covr2-checkpoint', str(weights), '--decode-workers', '4', '--video-batch-size', '8',
        '--covr2-frame-batch-size', '2']) == 0
    assert all(kw['decode_workers'] == 4 and kw['video_batch_size'] == 8 for kw in baseline)
    assert gallery[0]['decode_workers'] == 4 and gallery[0]['frame_batch_size'] == 2


@pytest.mark.parametrize('option,value', [('--decode-workers', '-1'), ('--video-batch-size', '0'),
                                         ('--covr2-frame-batch-size', '0')])
def test_pipeline_rejects_invalid_throughput_controls(option, value):
    from visual_retrieval.cli.webvid_covr import main
    with pytest.raises(SystemExit) as error:
        main([option, value, '--stage', 'check'])
    assert error.value.code == 2
