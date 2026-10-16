import base64
from io import BytesIO
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from visual_retrieval.demo.frames import captured_frame,image_key


def test_frame_source_time_hash_and_stale_preview_guards(tmp_path):
    image=Image.new('RGB',(4,4),'blue');f=BytesIO();image.save(f,format='PNG')
    item=dict(dataset='activitynet',source_video_id='source',start=10,end=12)
    ctx=dict(revision=2,page=1,frame_item=item)
    class Media:
        def video(self,row):return tmp_path/'preview.mp4'
    payload=dict(data='data:image/png;base64,'+base64.b64encode(f.getvalue()).decode(),
                 time=.7,src='http://localhost/file=/cache/preview.mp4',token='2:1',copied=True)
    actual,key,time,copied=captured_frame(json.dumps(payload),ctx,Media())
    assert time==10.7 and copied and key==image_key(np.asarray(actual))
    assert ctx['reference_sources'][key]==('activitynet','source')
    for changed in [dict(token='1:1'),dict(src='http://localhost/old.mp4'),dict(time=3)]:
        with pytest.raises(ValueError): captured_frame(json.dumps(dict(payload,**changed)),ctx,Media())


def test_composer_drafts_have_no_change_listener_and_route_only_on_submit(tmp_path):
    from visual_retrieval.demo.app import build_demo
    from test_covr2_service import composed_fixture
    service,_,_=composed_fixture(tmp_path)
    app=build_demo(service,None,'clip')
    config=app.get_config_file()
    query=next(c['id'] for c in config['components'] if c['type']=='multimodaltextbox')
    assert not any([query,'change'] in d['targets'] or (query,'change') in d['targets']
                   for d in config['dependencies'])
    f={v.fn.__name__:v.fn for v in app.fns.values() if v.fn}
    ctx=dict(revision=1,page=2,model='clip',dataset='all',text_model='clip',frame_item={})
    value=dict(text='make it run',files=['reference.png'])
    output=f['prepare_search'](value,'clip','all',ctx)
    assert output[0]['value']=='covr2' and ctx['model']=='covr2'
    assert ctx['revision']==2 and ctx['page']==0 and ctx['frame_item'] is None
    assert output[2]==value
    value['text']='changed after submit'
    assert output[2]['text']=='make it run'
    output=f['prepare_search'](dict(text='keep typing',files=[]),'covr2','all',ctx)
    assert output[0]['value']=='clip' and ctx['model']=='clip'


def test_submitting_image_without_covr2_setup_returns_actionable_message(tmp_path):
    from visual_retrieval.demo.app import build_demo
    from visual_retrieval.demo.service import SearchService
    from test_demo_service import saved_gallery,make_spec,Encoder
    saved_gallery(tmp_path/'a')
    service=SearchService([make_spec(tmp_path/'a','a')],'cpu',lambda *a:Encoder())
    app=build_demo(service,None,'clip')
    f={v.fn.__name__:v.fn for v in app.fns.values() if v.fn}
    image=tmp_path/'image.png';Image.new('RGB',(4,4)).save(image)
    ctx=dict(revision=0,selection=0,page=0,model='clip',dataset='all',text_model='clip')
    value=dict(text='make it run',files=[str(image)])
    prepared=f['prepare_search'](value,'clip','all',ctx)
    result=f['do_search'](prepared[2],'covr2','all',ctx)
    assert 'both CoVR-2 checkpoint options' in result[4]
