import json
from pathlib import Path
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image
import pytest


pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='FFmpeg is required for media integration tests')


def source_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
    assert writer.isOpened()
    for i in range(40):
        colour = (0, 0, 230) if i < 20 else (230, 0, 0)
        writer.write(np.full((48,64,3), colour, dtype=np.uint8))
    writer.release()


def result(path, start=2, end=3):
    return dict(path=str(path), start=start, end=end, item_id='clip', dataset='data')


def test_preview_is_browser_mp4_trimmed_to_segment_and_thumbnail_matches(tmp_path):
    from visual_retrieval.demo.media import MediaCache
    source = tmp_path / 'source.mp4'; source_video(source)
    cache = MediaCache(tmp_path / 'cache')
    thumb, video = cache.thumbnail(result(source)), cache.video(result(source))
    image = np.asarray(Image.open(thumb))
    assert image.shape == (270,480,3) and image[:,:,2].mean() > image[:,:,0].mean()
    meta = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name:format=duration','-of','json',str(video)]))
    assert meta['streams'][0]['codec_name'] == 'h264'
    assert float(meta['format']['duration']) == pytest.approx(1., abs=.1)
    cap = cv2.VideoCapture(str(video)); ok, first = cap.read(); cap.release()
    assert ok and first[:,:,0].mean() > first[:,:,2].mean()
    assert thumb.is_relative_to(cache.root) and video.is_relative_to(cache.root)
    assert not list(cache.root.rglob('*.tmp.*'))


def test_media_cache_reuses_files_and_invalidates_changed_source(tmp_path, monkeypatch):
    from visual_retrieval.demo.media import MediaCache
    source = tmp_path / 'source.mp4'; source_video(source)
    cache = MediaCache(tmp_path / 'cache')
    first = cache.thumbnail(result(source))
    original = subprocess.run
    def forbidden(*args, **kwargs):
        raise AssertionError('A cached thumbnail must not decode again')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    assert cache.thumbnail(result(source)) == first
    monkeypatch.setattr(subprocess, 'run', original)
    source.touch()
    assert cache.thumbnail(result(source)) != first
    assert cache.thumbnail(result(source, 0, 1)) != first


def test_concurrent_media_requests_share_one_complete_file(tmp_path):
    from visual_retrieval.demo.media import MediaCache
    source = tmp_path / 'source.mp4'; source_video(source)
    cache = MediaCache(tmp_path / 'cache')
    with ThreadPoolExecutor(2) as executor:
        files = list(executor.map(lambda _: cache.video(result(source)), range(2)))
    assert files[0] == files[1] and files[0].stat().st_size > 0


def test_missing_source_and_bad_interval_are_clear_errors(tmp_path):
    from visual_retrieval.demo.media import MediaCache, MediaError
    cache = MediaCache(tmp_path / 'cache')
    with pytest.raises(MediaError, match='unavailable'):
        cache.video(result(tmp_path / 'missing.mp4'))
    source = tmp_path / 'source.mp4'; source_video(source)
    for start, end in [(2,1), (None,2), (float('nan'),3), (-1,1)]:
        with pytest.raises(MediaError, match='interval'):
            cache.thumbnail(result(source,start,end))
    assert Image.open(cache.placeholder()).size == (480,270)


def test_whole_video_and_paths_with_spaces(tmp_path):
    from visual_retrieval.demo.media import MediaCache
    source = tmp_path / 'video with spaces.mp4'; source_video(source)
    cache = MediaCache(tmp_path / 'cache with spaces')
    assert cache.thumbnail(result(source,None,None)).is_file()
    video = cache.video(result(source,None,None))
    duration = float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(video)]))
    assert duration == pytest.approx(4., abs=.1)
