import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_demo_service import Encoder, make_spec, saved_gallery


def test_sharing_requires_credentials_and_local_launch_can_be_private():
    from visual_retrieval.demo.app import auth_from_env
    assert auth_from_env(False, {}) is None
    with pytest.raises(ValueError, match='VR_DEMO_USERNAME'):
        auth_from_env(True, {})
    with pytest.raises(ValueError, match='both'):
        auth_from_env(False, {'VR_DEMO_USERNAME':'user'})
    assert auth_from_env(True, {'VR_DEMO_USERNAME':'user', 'VR_DEMO_PASSWORD':'password'}) == ('user','password')


def test_controller_handles_missing_thumbnails_and_uses_canonical_media_paths(tmp_path):
    from visual_retrieval.demo.app import DemoController
    from visual_retrieval.demo.service import SearchService
    from visual_retrieval.demo.media import MediaError
    saved_gallery(tmp_path / 'a')
    service = SearchService([make_spec(tmp_path / 'a', 'a')], device='cpu', encoder_loader=lambda *a: Encoder())
    class Media:
        def thumbnail(self, item):
            raise MediaError('missing')
        def placeholder(self):
            return tmp_path / 'placeholder.jpg'
        def video(self, item):
            assert item['path'] == str((tmp_path / 'a/raw_videos/same.mp4').resolve())
            return tmp_path / 'preview.mp4'
    controller = DemoController(service, Media())
    images, results, status = controller.search('alpha', 'clip', 'all')
    assert images[0][0] == str(tmp_path / 'placeholder.jpg') and 'unavailable' in status
    results[0]['path'] = '/private.txt'
    video, description = controller.select(results, 0)
    assert video == str(tmp_path / 'preview.mp4') and 'same' in description
    with pytest.raises(ValueError, match='Select'):
        controller.select(results, 10)


def test_ui_has_minimal_controls_and_default_combined_gallery(tmp_path):
    pytest.importorskip('gradio')
    from visual_retrieval.demo.app import build_demo
    from visual_retrieval.demo.service import SearchService
    saved_gallery(tmp_path / 'a')
    service = SearchService([make_spec(tmp_path / 'a', 'a')], device='cpu', encoder_loader=lambda *a: Encoder())
    demo = build_demo(service, None, default_model='clip')
    config = demo.get_config_file()
    types = [c['type'] for c in config['components']]
    assert types.count('textbox') == 1 and types.count('button') == 1
    assert types.count('dropdown') == 2 and types.count('gallery') == 1 and types.count('video') == 1
    dataset = next(c for c in config['components'] if c['props'].get('elem_id') == 'dataset-choice')
    assert dataset['props']['value'] == 'all'
    assert any(d['api_name'] == 'search' for d in config['dependencies'])


def test_legacy_precision_is_visible_in_search_summary(tmp_path):
    from visual_retrieval.demo.app import DemoController
    from visual_retrieval.demo.service import SearchService
    folder = saved_gallery(tmp_path / 'a')
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest.pop('effective_precision')
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    service = SearchService([make_spec(tmp_path / 'a', 'a')], 'cpu', lambda *a: Encoder())
    assert 'effective precision not recorded' in DemoController(service, None).summary('clip')


def test_demo_cli_help_and_preflight_do_not_load_a_model(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/demo.py'
    subprocess.run([sys.executable, str(script), '--help'], check=True, cwd=tmp_path, capture_output=True)
    saved_gallery(tmp_path / 'a')
    config = tmp_path / 'demo.yaml'
    config.write_text(f'default_model: clip\ndatasets:\n  a:\n    root: {tmp_path / "a"}\n    features:\n      clip: features/clip\n')
    # Checkpoint path is deliberately nonexistent. A GPU/model load would fail.
    run = subprocess.run([sys.executable, str(script), '--config', str(config), '--check'],
                         check=True, cwd=tmp_path, capture_output=True, text=True)
    assert 'clip' in run.stdout and '1 clips' in run.stdout


def test_selector_changes_discard_pending_search_and_preview(tmp_path):
    gr = pytest.importorskip('gradio')
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from visual_retrieval.demo.app import build_demo
    from visual_retrieval.demo.service import SearchService
    saved_gallery(tmp_path / 'a')
    entered, release = threading.Event(), threading.Event()
    class Media:
        def thumbnail(self, item):
            entered.set(); assert release.wait(10)
            return tmp_path / 'thumb.jpg'
        def video(self, item):
            entered.set(); assert release.wait(10)
            return tmp_path / 'preview.mp4'
    service = SearchService([make_spec(tmp_path / 'a', 'a')], 'cpu', lambda *a: Encoder())
    demo = build_demo(service, Media(), 'clip')
    funcs = {f.fn.__name__: f.fn for f in demo.fns.values() if f.fn}
    context = dict(revision=0, selection=0, model='clip', dataset='all')
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(funcs['do_search'], 'alpha', 'clip', 'all', context)
        assert entered.wait(5)
        funcs['change_dataset']('clip', 'a', context)
        release.set()
        assert all(x == gr.skip() for x in pending.result())
        entered.clear(); release.clear()
        context.update(dataset='all')
        rows = service.search('alpha', 'clip')
        rows[0]['view_revision'] = context['revision']
        event = gr.SelectData(None, {'index':0, 'value':None})
        pending = pool.submit(funcs['select_video'], rows, context, event)
        assert entered.wait(5)
        funcs['change_dataset']('clip', 'a', context)
        release.set()
        assert all(x == gr.skip() for x in pending.result())
