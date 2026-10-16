import gc
import json
import weakref
import numpy as np
from PIL import Image
import pytest
from test_demo_service import saved_gallery, make_spec, Encoder
from visual_retrieval.demo.service import SearchService
from visual_retrieval.demo.composed import ComposedSearchService, composer_value
from visual_retrieval.composed.gallery import FORMAT, digest, token_path


def tokens():
    x=np.zeros((15,32,256),dtype=np.float32);x[:,:,0]=1
    return x


def composed_fixture(tmp_path, loader=None):
    saved_gallery(tmp_path/'a')
    spec=make_spec(tmp_path/'a','a')
    base,query=tmp_path/'base.pt',tmp_path/'query.pt'
    base.write_bytes(b'base');query.write_bytes(b'query')
    folder=spec.root/'features/covr2';(folder/'tokens').mkdir(parents=True)
    rows=[dict(item_id='same',source_video_id='same',path='same.mp4',start=None,end=None)]
    manifest=dict(identity=dict(format=FORMAT,base_sha256=digest(base),frames=15,precision='fp32'),gallery_items=rows)
    (folder/'manifest.json').write_text(json.dumps(manifest))
    np.save(token_path(folder,'same'),tokens())
    text=SearchService([spec],'cpu',lambda *a:Encoder())
    return ComposedSearchService(text,[spec],base,query,encoder_loader=loader),spec,folder


def test_engine_switch_releases_previous_gpu_owner_and_excludes_source(tmp_path):
    refs=[]
    class Composed:
        def __init__(self,*args,**kwargs):
            assert service.text._encoder is None
            refs.append(weakref.ref(self))
        def encode_query(self,image,text):return tokens()[0],np.eye(256,dtype=np.float32)[0]
    service,_,_=composed_fixture(tmp_path,Composed)
    service.search('alpha','clip')
    old=weakref.ref(service.text._encoder)
    assert len(service.search_composed(np.zeros((4,4,3)), 'make it run'))==1
    gc.collect();assert old() is None
    assert service.search_composed(np.zeros((4,4,3)), 'run',source=('a','same'))==[]
    service.search('alpha','clip');gc.collect();assert refs[0]() is None


def test_canonical_composed_resolution_and_incompatible_cache_rejection(tmp_path):
    service,spec,folder=composed_fixture(tmp_path)
    row=dict(model='covr2',dataset='a',item_id='same',path='/secret')
    assert service.resolve_result(row)['path']==str((spec.root/'raw_videos/same.mp4').resolve())
    manifest=json.loads((folder/'manifest.json').read_text());manifest['identity']['base_sha256']='wrong'
    (folder/'manifest.json').write_text(json.dumps(manifest))
    invalid=ComposedSearchService(service.text,[spec],service.base,service.checkpoint)
    assert not invalid.composed_available and any('provenance' in e for e in invalid.issues)


def test_composer_accepts_one_still_image_and_rejects_video_multiple_and_animation(tmp_path):
    p=tmp_path/'image.png';Image.new('RGB',(8,8)).save(p)
    assert composer_value({'text':'make it run','files':[str(p)]})[1].shape==(8,8,3)
    for files in [[str(p),str(p)],[str(tmp_path/'video.mp4')]]:
        with pytest.raises(ValueError): composer_value({'text':'run','files':files})
    gif=tmp_path/'image.gif';Image.new('RGB',(8,8)).save(gif)
    with pytest.raises(ValueError): composer_value({'files':[str(gif)]})


def test_corrupt_item_is_omitted_without_disabling_other_items(tmp_path):
    class Composed:
        def __init__(self,*a,**k):pass
        def encode_query(self,*a):return tokens()[0],np.eye(256,dtype=np.float32)[0]
    service,spec,folder=composed_fixture(tmp_path,Composed)
    manifest=json.loads((folder/'manifest.json').read_text())
    row=dict(manifest['gallery_items'][0],item_id='broken')
    manifest['gallery_items'].append(row);(folder/'manifest.json').write_text(json.dumps(manifest))
    token_path(folder,'broken').write_bytes(b'')
    valid=ComposedSearchService(service.text,[spec],service.base,service.checkpoint,Composed)
    assert valid.composed_available and valid.gallery_counts('covr2')=={'a':1}
    assert valid.omitted_counts('covr2')=={'a':1}
    assert len(valid.search_composed(np.zeros((4,4,3)),'run'))==1
    token_path(folder,'same').write_bytes(b'corrupted after startup')
    assert valid.search_composed(np.zeros((4,4,3)),'run')==[]
    assert valid.omitted_counts('covr2')=={'a':2}
