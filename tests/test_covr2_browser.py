"""Browser check: copy paused frame, paste, composed ranking and direct fallback."""
import json
import os
import socket
import subprocess
import shutil
import numpy as np
import pytest
from test_demo_service import saved_gallery,make_spec,Encoder
from test_demo_media import source_video
from test_covr2_service import tokens


@pytest.mark.skipif(os.environ.get('VR_BROWSER_TEST')!='1',reason='Opt-in Playwright integration')
def test_copy_paste_composed_search_and_use_frame_fallback(tmp_path,monkeypatch):
    from playwright.sync_api import sync_playwright
    from visual_retrieval.demo.service import SearchService
    from visual_retrieval.demo.composed import ComposedSearchService
    from visual_retrieval.demo.media import MediaCache
    from visual_retrieval.demo.app import build_demo,CSS
    from visual_retrieval.composed.gallery import FORMAT,digest,token_path
    monkeypatch.setenv('NO_PROXY','127.0.0.1,localhost');monkeypatch.setenv('no_proxy','127.0.0.1,localhost')
    root=tmp_path/'dataset';ids=tuple(f'video{i:02d}' for i in range(25))
    folder=saved_gallery(root,ids=ids,features=[[1,0]]*25)
    for path in (root/'raw_videos').glob('*.mp4'): source_video(path)
    spec=make_spec(root,'a');base=tmp_path/'base.pt';query=tmp_path/'query.pt'
    base.write_bytes(b'base');query.write_bytes(b'query')
    gallery=root/'features/covr2';(gallery/'tokens').mkdir(parents=True)
    rows=json.loads((folder/'manifest.json').read_text())['gallery_items']
    (gallery/'manifest.json').write_text(json.dumps(dict(identity=dict(format=FORMAT,base_sha256=digest(base),frames=15,precision='fp32'),gallery_items=rows)))
    for id in ids: np.save(token_path(gallery,id),tokens())
    seen=[]
    class Composed:
        def __init__(self,*a,**k):pass
        def encode_query(self,image,text):
            seen.append((image.copy(),text));return tokens()[0],np.eye(256,dtype=np.float32)[0]
    service=ComposedSearchService(SearchService([spec],'cpu',lambda *a:Encoder()),[spec],base,query,Composed)
    # Playwright's bundled Chromium lacks H.264; use a decoded VP9 preview
    # for canvas/clipboard checks. Production H.264 serving has its own tests.
    class BrowserMedia(MediaCache):
        def video(self,item):
            original=super().video(item); webm=original.with_suffix('.webm')
            if not webm.exists():
                subprocess.run(['ffmpeg','-v','error','-i',str(original),'-c:v','libvpx-vp9','-an',str(webm)],check=True)
            return webm
    media=BrowserMedia(tmp_path/'cache');app=build_demo(service,media,'clip',page_size=20)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    app.launch(server_name='127.0.0.1',server_port=port,prevent_thread_lock=True,
               allowed_paths=[str(media.root)],blocked_paths=[str(root/'raw_videos')],css=CSS,footer_links=[])
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            context=browser.new_context(permissions=['clipboard-read','clipboard-write'])
            page=context.new_page();page.goto(f'http://127.0.0.1:{port}')
            page.locator('#query textarea').fill('a person');page.locator('#search-button').click()
            page.wait_for_function("document.querySelectorAll('#results img').length===20")
            page.evaluate("window.previousResults=Array.from(document.querySelectorAll('#results img')); window.previousSummary=document.querySelector('#summary').innerText")
            page.locator('#query textarea').press_sequentially(' draft',delay=30)
            page.wait_for_timeout(400)
            assert page.evaluate("previousResults.every((img,i)=>img===document.querySelectorAll('#results img')[i]) && previousSummary===document.querySelector('#summary').innerText")
            page.locator('#results img').first.click();page.locator('video').wait_for(state='visible')
            page.wait_for_function("document.querySelector('video').readyState>=2")
            page.locator('video').evaluate('(v)=>{v.pause();v.currentTime=2.5;}')
            page.wait_for_function("Math.abs(document.querySelector('video').currentTime-2.5)<.05 && !document.querySelector('video').seeking")
            page.locator('#copy-frame').click()
            page.wait_for_function("document.querySelector('#frame-status').innerText.includes('Copied frame at 2.50s')")
            page.locator('#query textarea').fill('make the person run')
            page.locator('#query textarea').press('Control+V')
            page.locator('#query .thumbnail-image').wait_for()
            preview=page.locator('#query .thumbnail-image')
            assert preview.bounding_box()['width']>=280
            assert preview.bounding_box()['height']>=180
            assert preview.evaluate("img=>getComputedStyle(img).objectFit==='contain'")
            page.screenshot(path=str(tmp_path/'reference-desktop.png'),full_page=True)
            page.set_viewport_size({'width':390,'height':844})
            page.wait_for_timeout(200)
            box=preview.bounding_box()
            page.screenshot(path=str(tmp_path/'reference-mobile.png'),full_page=True)
            assert box['width']>=240 and box['height']>=180
            assert box['x']+box['width']<=390
            assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
            assert page.get_by_role('button',name='Remove file',exact=True).is_visible()
            page.set_viewport_size({'width':1280,'height':900})
            assert page.locator('#model-choice input').input_value()=='CLIP'
            assert page.locator('#results img').count()==20
            assert page.locator('#query textarea').input_value()=='make the person run'
            page.locator('#search-button').click()
            page.wait_for_function("document.querySelector('#model-choice input').value==='CoVR-2'")
            page.wait_for_function("document.querySelector('#summary').innerText.includes('24 results')")
            assert len(seen)==1 and seen[0][1]=='make the person run'
            # Late frame is blue, first frame is red: copying a thumbnail would fail.
            assert seen[0][0][:,:,2].mean()>seen[0][0][:,:,0].mean()
            assert 'video00' not in page.locator('#results').inner_text()
            page.locator('#next-page').click()
            page.wait_for_function("document.querySelectorAll('#results img').length===4")
            page.wait_for_function("document.querySelector('#results .caption-label')?.innerText.startsWith('21')")
            page.locator('#results img').first.click();page.locator('video').wait_for(state='visible')
            page.wait_for_function("document.querySelector('video').readyState>=2")
            # Direct attachment works even when clipboard write is unavailable.
            page.evaluate("Object.defineProperty(navigator.clipboard,'write',{value:async()=>{throw new Error('blocked')}})")
            page.locator('#copy-frame').click()
            page.wait_for_function("document.querySelector('#frame-status').innerText.includes('blocked')")
            page.locator('#use-frame').click()
            page.wait_for_function("document.querySelector('#frame-status').innerText.includes('attached')")
            assert page.locator('#query textarea').input_value()=='make the person run'
            page.screenshot(path=str(tmp_path/'composed-search.png'),full_page=True)
            page.get_by_role('button',name='Remove file',exact=True).click()
            assert page.locator('#model-choice input').input_value()=='CoVR-2'
            assert page.locator('#results img').count()==4
            assert page.locator('#query textarea').input_value()=='make the person run'
            page.locator('#query textarea').press('Enter')
            page.wait_for_function("document.querySelector('#model-choice input').value==='CLIP'")
            page.wait_for_function("document.querySelector('#summary').innerText.includes('25 results')")
            browser.close()
    finally:app.close();service.close()
