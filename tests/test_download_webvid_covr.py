import csv
import json
from pathlib import Path

import pytest

from download_data import download_webvid_covr as downloader


def rows():
    return [dict(pth1='1/101',pth2='2/202',edit='make it blue',txt1='red',txt2='blue'),
            dict(pth1='1/101',pth2='3/303',edit='make it green',txt1='red',txt2='green')]


def setup_download(tmp_path, monkeypatch, bad_video=None):
    annotations=tmp_path/'annotations.csv'
    with annotations.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows()[0]));writer.writeheader();writer.writerows(rows())
    calls=[]
    def download(url,destination,**kwargs):
        calls.append(url)
        Path(destination).parent.mkdir(parents=True,exist_ok=True)
        if bad_video and bad_video in url: raise OSError('unavailable')
        Path(destination).write_bytes(b'valid video')
    monkeypatch.setattr(downloader,'download_file',download)
    def probe(path,**kwargs):
        if Path(path).read_bytes()!=b'valid video': raise ValueError('not a video')
        return 4.0
    monkeypatch.setattr(downloader,'probe_video',probe)
    monkeypatch.setattr(downloader,'save_reference_frame',lambda video,dest,duration: Path(dest).write_bytes(b'jpg'))
    return annotations,calls


def test_sample_downloads_both_videos_and_preview_and_resumes(tmp_path,monkeypatch):
    annotations,calls=setup_download(tmp_path,monkeypatch)
    root=tmp_path/'dataset'
    result=downloader.prepare(root,samples=1,annotation_file=annotations,source='hub')
    assert len(result['samples'])==1
    sample=result['samples'][0]
    assert sample['modification']=='make it blue'
    for key in ('reference_video','target_video','reference_image'):
        assert (root/sample[key]).is_file()
    assert len(calls)==2
    preview=root/'samples/test/preview.html'
    assert 'make it blue' in preview.read_text() and '<video' in preview.read_text()
    downloader.prepare(root,samples=2,annotation_file=annotations,source='hub')
    assert len(calls)==3  # cached reference/target reused; only one new video
    stored=json.loads((root/'samples/test/subset.json').read_text())
    assert len(stored['samples'])==2
    assert (root/stored['samples'][0]['reference_video']).name=='101.mp4'


def test_failed_target_does_not_count_as_a_complete_sample(tmp_path,monkeypatch):
    annotations,calls=setup_download(tmp_path,monkeypatch,bad_video='202.mp4')
    root=tmp_path/'dataset'
    result=downloader.prepare(root,samples=1,annotation_file=annotations,source='hub')
    assert len(result['samples'])==1
    assert result['samples'][0]['target_id']=='303'
    assert len(result['failures'])==1


def test_corrupt_cache_is_redownloaded_and_shortfall_is_explicit(tmp_path,monkeypatch):
    annotations,calls=setup_download(tmp_path,monkeypatch,bad_video='202.mp4')
    root=tmp_path/'dataset';(root/'raw_videos').mkdir(parents=True)
    (root/'raw_videos/101.mp4').write_text('<html>403</html>')
    result=downloader.prepare(root,samples=5,annotation_file=annotations,source='hub',max_attempts=2)
    assert not result['complete'] and result['requested_samples']==5
    assert (root/'raw_videos/101.mp4').read_bytes()==b'valid video'
    assert len(result['samples'])==1


def test_annotation_paths_cannot_escape_output_and_empty_edits_rejected():
    for path in ('../../secret','1/../../x','https://host/file','1/x.mp4','-1/101'):
        with pytest.raises(ValueError):downloader.video_path(path)
    assert downloader.video_path('112/1016223889')=='112/1016223889.mp4'
    with pytest.raises(ValueError):downloader.validate_row(dict(rows()[0],edit='  '))


def test_html_preview_escapes_annotation_text(tmp_path,monkeypatch):
    annotations,calls=setup_download(tmp_path,monkeypatch)
    text=annotations.read_text().replace('make it blue','<script>alert(1)</script>')
    annotations.write_text(text)
    root=tmp_path/'dataset';downloader.prepare(root,samples=1,annotation_file=annotations,source='hub')
    html=(root/'samples/test/preview.html').read_text()
    assert '<script>' not in html and '&lt;script&gt;' in html


def test_failed_transfer_keeps_existing_file_and_removes_partial(tmp_path,monkeypatch):
    import requests
    class Response:
        headers={}
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def raise_for_status(self):pass
        def iter_content(self,chunk_size):
            yield b'partial'
            raise requests.ConnectionError('connection dropped')
    monkeypatch.setattr(downloader.requests,'get',lambda *a,**kw:Response())
    destination=tmp_path/'video.mp4';destination.write_bytes(b'previous complete file')
    with pytest.raises(OSError,match='connection dropped'):
        downloader.download_file('https://example.org/video.mp4',destination,retries=0)
    assert destination.read_bytes()==b'previous complete file'
    assert not list(tmp_path.glob('*.part'))


def test_ctrl_c_preserves_completed_pairs_and_allows_resume(tmp_path,monkeypatch):
    annotations,calls=setup_download(tmp_path,monkeypatch)
    original=downloader.download_file
    def interrupted(url,destination,**kwargs):
        if '303.mp4' in url: raise KeyboardInterrupt
        return original(url,destination,**kwargs)
    monkeypatch.setattr(downloader,'download_file',interrupted)
    root=tmp_path/'dataset'
    with pytest.raises(KeyboardInterrupt):
        downloader.prepare(root,samples=2,annotation_file=annotations,source='hub')
    result=json.loads((root/'samples/test/subset.json').read_text())
    assert len(result['samples'])==1
    monkeypatch.setattr(downloader,'download_file',original)
    result=downloader.prepare(root,samples=2,annotation_file=annotations,source='hub')
    assert result['complete'] and len(calls)==3


def test_original_url_fallback_keeps_same_pair_identity(tmp_path,monkeypatch):
    annotations,calls=setup_download(tmp_path,monkeypatch)
    def download(url,destination,**kwargs):
        calls.append(url)
        if 'huggingface.co' in url: raise OSError('missing from mirror')
        if url.endswith('.json'):
            Path(destination).write_text(json.dumps({'1/101.mp4':'https://original.test/101.mp4',
                '2/202.mp4':'https://original.test/202.mp4'}))
        else:Path(destination).write_bytes(b'valid video')
    monkeypatch.setattr(downloader,'download_file',download)
    result=downloader.prepare(tmp_path/'dataset',samples=1,annotation_file=annotations)
    assert result['complete'] and result['samples'][0]['target_id']=='202'
    assert len([c for c in calls if c.endswith('.json')])==1


def test_nonpositive_sample_count_fails_before_network(tmp_path):
    with pytest.raises(ValueError,match='positive'):
        downloader.prepare(tmp_path,samples=0)


def test_proxy_failure_is_reported_as_a_download_error(tmp_path,monkeypatch):
    import requests
    calls=[]
    def unavailable(*a,**kw):
        calls.append(1)
        raise requests.exceptions.ProxyError('proxy denied the connection')
    monkeypatch.setattr(downloader.requests,'get',unavailable)
    with pytest.raises(OSError,match='proxy'):
        downloader.download_file('https://example.org/video.mp4',tmp_path/'video.mp4',retries=0)
    assert len(calls)==1


@pytest.mark.skipif(not downloader.shutil.which('ffmpeg') or not downloader.shutil.which('ffprobe'),reason='FFmpeg required')
def test_five_sample_http_download_with_real_video_and_reference_decoding(tmp_path,monkeypatch):
    from functools import partial
    from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
    import subprocess
    import threading
    media=tmp_path/'http';media.mkdir()
    video=media/'fixture.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=64x48:r=5:d=1',
                    '-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
    blob=video.read_bytes()
    annotation=tmp_path/'five.csv'
    with annotation.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['pth1','pth2','edit','txt1','txt2']);writer.writeheader()
        for i in range(5):
            for video_id in (100+i,200+i):(media/f'{video_id}.mp4').write_bytes(blob)
            writer.writerow(dict(pth1=f'0/{100+i}',pth2=f'1/{200+i}',edit=f'change {i}',txt1='reference',txt2='target'))
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(media)))
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    monkeypatch.setenv('NO_PROXY','127.0.0.1');monkeypatch.setenv('no_proxy','127.0.0.1')
    monkeypatch.setattr(downloader,'hub_url',lambda repo,revision,path:f'http://127.0.0.1:{server.server_port}/{Path(path).name}')
    root=tmp_path/'dataset'
    try:
        result=downloader.prepare(root,samples=5,annotation_file=annotation,source='hub')
        assert result['complete'] and len(result['samples'])==5 and result['unique_videos']==10
        assert len(list((root/'raw_videos').glob('*.mp4')))==10
        for sample in result['samples']:
            frame=root/sample['reference_image']
            assert frame.read_bytes().startswith(b'\xff\xd8') and frame.stat().st_size>100
        assert (root/'samples/test/preview.html').read_text().count('<video ')==5
        assert downloader.main(['--root',str(root),'--samples','5','--annotation-file',str(annotation),'--source','hub'])==0
    finally:
        server.shutdown();server.server_close();worker.join()


@pytest.mark.skipif(not downloader.shutil.which('ffmpeg') or not downloader.shutil.which('ffprobe'),reason='FFmpeg required')
def test_valid_headers_do_not_make_truncated_video_acceptable(tmp_path):
    import subprocess
    valid=tmp_path/'valid.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=128x96:r=10:d=4',
                    '-c:v','libx264','-preset','ultrafast','-movflags','+faststart',str(valid)],check=True)
    broken=tmp_path/'broken.mp4';blob=valid.read_bytes();broken.write_bytes(blob[:len(blob)//3])
    assert downloader.probe_video(valid)==4.0
    with pytest.raises((ValueError,OSError,subprocess.SubprocessError)):
        downloader.probe_video(broken)
