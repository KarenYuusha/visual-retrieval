import json
import numpy as np
from test_demo_media import source_video
from test_covr2_service import tokens
from visual_retrieval.composed.prepare import prepare
from visual_retrieval.composed.gallery import token_path


def test_segment_extraction_uses_15_frames_and_resumes_without_loading_model(tmp_path):
    root=tmp_path/'dataset';(root/'raw_videos').mkdir(parents=True)
    source_video(root/'raw_videos/source.mp4')
    manifest=tmp_path/'source.json';manifest.write_text(json.dumps(dict(
        protocol='combined_subset_activitynet_segments',gallery_items=[dict(
            item_id='seg',source_video_id='source',path='source.mp4',start=2,end=3)])))
    base=tmp_path/'base.pt';base.write_bytes(b'base')
    calls=[]
    class Encoder:
        def __init__(self,*a,**k): calls.append('load')
        def encode_frames(self,frames):
            assert frames.shape==(15,48,64,3)
            assert frames[:,:,:,2].mean()>frames[:,:,:,0].mean()
            calls.append('encode');return tokens()
    prepare(root,manifest,base,device='cpu',encoder_factory=Encoder)
    prepare(root,manifest,base,device='cpu',encoder_factory=Encoder)
    assert calls==['load','encode']
    assert np.load(token_path(root/'features/covr2','seg')).shape==(15,32,256)
    # Changed segment bounds invalidate its existing tokens instead of reusing them.
    source=json.loads(manifest.read_text());source['gallery_items'][0].update(start=2.2,end=2.8)
    manifest.write_text(json.dumps(source))
    prepare(root,manifest,base,device='cpu',encoder_factory=Encoder)
    assert calls==['load','encode','load','encode']
    token_path(root/'features/covr2','seg').write_bytes(b'')
    prepare(root,manifest,base,device='cpu',encoder_factory=Encoder)
    assert calls==['load','encode','load','encode','load','encode']
