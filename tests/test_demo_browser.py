"""Opt-in browser integration: VR_BROWSER_TEST=1 pytest tests/test_demo_browser.py.

Install Playwright and its Chromium first. H.264 decoding is checked when the
browser supports it; media delivery/FFprobe validation always runs.
"""
import json
import os
from pathlib import Path
import socket
import subprocess

import pytest

from test_demo_service import Encoder, make_spec, saved_gallery
from test_demo_media import source_video


@pytest.mark.skipif(os.environ.get('VR_BROWSER_TEST') != '1', reason='Opt-in Playwright integration')
def test_authenticated_browser_search_filters_and_segment_delivery(tmp_path, monkeypatch):
    pytest.importorskip('gradio')
    playwright = pytest.importorskip('playwright.sync_api')
    from visual_retrieval.demo.service import SearchService
    from visual_retrieval.demo.media import MediaCache
    from visual_retrieval.demo.app import build_demo, CSS
    monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
    monkeypatch.setenv('no_proxy', '127.0.0.1,localhost')
    specs = []
    for key in ('msrvtt', 'activitynet'):
        root = tmp_path / key
        for model in ('clip', 'clip4clip', 'internvideo2'):
            folder = saved_gallery(root, model, ids=tuple(f'video{i}' for i in range(5)),
                                   features=[[1,0],[.9,.1],[.8,.2],[.7,.3],[.6,.4]])
            if key == 'activitynet':
                manifest = json.loads((folder / 'manifest.json').read_text())
                manifest['protocol'] = 'combined_subset_activitynet_segments'
                for row in manifest['gallery_items']:
                    row.update(start=1, end=2)
                (folder / 'manifest.json').write_text(json.dumps(manifest))
        for video in (root / 'raw_videos').glob('*.mp4'):
            source_video(video)
        specs.append(make_spec(root, key, ('clip', 'clip4clip', 'internvideo2')))
    service = SearchService(specs, 'cpu', lambda *a: Encoder())
    media = MediaCache(tmp_path / 'cache')
    app = build_demo(service, media)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    app.launch(server_name='127.0.0.1', server_port=port, prevent_thread_lock=True,
               auth=('demo', 'test-password'), allowed_paths=[str(media.root)],
               blocked_paths=[str(s.root / 'raw_videos') for s in specs], css=CSS, footer_links=[])
    try:
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width':1280, 'height':1000})
            url = f'http://127.0.0.1:{port}'
            assert page.request.get(url + '/gradio_api/file=' + str(media.placeholder())).status == 401
            page.goto(url, wait_until='domcontentloaded')
            page.locator('input').nth(0).fill('demo')
            page.locator('input').nth(1).fill('test-password')
            page.get_by_role('button', name='Login', exact=True).click()
            page.locator('#query textarea').fill('alpha')
            page.locator('#search-button').click()
            page.wait_for_function("document.querySelectorAll('#results img').length === 10")
            page.locator('#results img').nth(1).click()
            page.locator('video').wait_for(state='visible')
            assert '1.00–2.00s' in page.locator('#selection-detail').inner_text()
            src = page.locator('video').evaluate('(v)=>v.currentSrc')
            response = page.request.get(src)
            assert response.status == 200
            delivered = tmp_path / 'delivered.mp4'; delivered.write_bytes(response.body())
            duration = float(subprocess.check_output(['ffprobe','-v','error','-show_entries',
                'format=duration','-of','default=noprint_wrappers=1:nokey=1',str(delivered)]))
            assert .9 < duration < 1.2
            source = specs[0].root / 'raw_videos/video0.mp4'
            assert page.request.get(url + '/gradio_api/file=' + str(source)).status == 403
            if page.locator('video').evaluate("(v)=>v.canPlayType('video/mp4; codecs=\"avc1.42E01E\"')"):
                page.locator('video').evaluate('(v)=>v.play()')
                page.wait_for_function('document.querySelector("video").readyState >= 2')
            page.locator('#dataset-choice').click()
            page.get_by_role('option', name='MSRVTT', exact=True).click()
            page.wait_for_function("document.querySelector('#summary').innerText.includes('1 dataset')")
            page.wait_for_timeout(500)  # Allow selector cancellation to finish before the next request.
            page.locator('#query textarea').press('Enter')
            page.wait_for_function("document.querySelectorAll('#results img').length === 5")
            page.locator('#model-choice').click()
            page.get_by_role('option', name='CLIP', exact=True).click()
            page.wait_for_function("document.querySelector('#summary').innerText.includes('2 datasets')")
            page.wait_for_timeout(500)
            page.locator('#search-button').click()
            page.wait_for_function("document.querySelectorAll('#results img').length === 10")
            browser.close()
    finally:
        app.close(); service.close()
