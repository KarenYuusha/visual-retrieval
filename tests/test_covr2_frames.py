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


def test_attachment_switch_preserves_prior_text_model_and_invalidates_old_results(tmp_path):
    from visual_retrieval.demo.app import build_demo
    from test_covr2_service import composed_fixture
    service,_,_=composed_fixture(tmp_path)
    app=build_demo(service,None,'clip')
    f={v.fn.__name__:v.fn for v in app.fns.values() if v.fn}
    ctx=dict(revision=1,page=2,model='clip',dataset='a',text_model='clip',frame_item={})
    value=dict(text='make it run',files=['reference.png'])
    output=f['change_attachment'](value,ctx)
    assert output[0]['value']=='covr2' and ctx['model']=='covr2'
    assert ctx['revision']==2 and ctx['page']==0 and ctx['frame_item'] is None
    same=f['change_attachment'](dict(value,text='keep typing'),ctx)
    import gradio as gr
    assert all(x==gr.skip() for x in same)
    output=f['change_attachment'](dict(text='keep typing',files=[]),ctx)
    assert output[0]['value']=='clip' and ctx['model']=='clip'
