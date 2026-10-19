import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from visual_retrieval.data.webvid_covr import load_subset


def make_subset(root):
    (root/'raw_videos').mkdir(parents=True)
    (root/'reference_frames').mkdir()
    folder=root/'samples/test';folder.mkdir(parents=True)
    for id in ('100','200','201'):
        (root/f'raw_videos/{id}.mp4').write_bytes(b'video')
        Image.new('RGB',(8,8),'blue').save(root/f'reference_frames/{id}.jpg')
    samples=[]
    for row,ref,target,text in [(7,'100','200','blue'),(8,'200','201','green'),(9,'100','200','green')]:
        samples.append(dict(row_index=row,reference_id=ref,target_id=target,
            reference_video=f'raw_videos/{ref}.mp4',target_video=f'raw_videos/{target}.mp4',
            reference_image=f'reference_frames/{ref}.jpg',modification=text,
            reference_caption='reference',target_caption='target'))
    path=folder/'subset.json';path.write_text(json.dumps(dict(dataset='WebVid-CoVR',split='test',
        complete=True,requested_samples=3,samples=samples)))
    return path


def test_subset_deduplicates_gallery_keeps_queries_and_rejects_escape(tmp_path):
    path=make_subset(tmp_path)
    data=load_subset(tmp_path)
    assert [r['item_id'] for r in data.items]==['100','200','201']
    assert [q['video_id'] for q in data.queries]==['200','201','200']
    assert len({q['query_id'] for q in data.queries})==3
    assert data.candidate_ids('targets')==['200','201']
    broken=json.loads(path.read_text());broken['samples'][0]['reference_image']='../../outside.jpg'
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='root|relative'):load_subset(tmp_path)


def test_subset_rejects_duplicate_query_ids_and_self_target(tmp_path):
    path=make_subset(tmp_path);doc=json.loads(path.read_text())
    doc['samples'][1]['row_index']=7;path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='Duplicate'):load_subset(tmp_path)
    doc['samples'][1]['row_index']=8
    doc['samples'][0]['target_id']='100';doc['samples'][0]['target_video']='raw_videos/100.mp4'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='reference.*target|self'):load_subset(tmp_path)


def test_baseline_encodes_every_video_and_text_without_evaluation_and_reuses_cache(tmp_path,monkeypatch):
    from visual_retrieval.features import webvid_covr as extraction
    make_subset(tmp_path);data=load_subset(tmp_path);checkpoint=tmp_path/'weights.pt';checkpoint.write_bytes(b'weights')
    calls=[]
    class Encoder:
        effective_precision='fp32'
        def encode_videos(self,clips):
            calls.append(('video',len(clips[0])))
            return np.ones((len(clips),512),np.float32)
        def encode_text(self,texts):
            calls.append(('text',len(texts)))
            return np.ones((len(texts),512),np.float32)
    monkeypatch.setattr(extraction,'read_video_frames',lambda path,count_frames:(np.zeros((count_frames,8,8,3),np.uint8),list(range(count_frames)),count_frames))
    output=tmp_path/'features/clip'
    kwargs=dict(checkpoint=checkpoint,device='cpu',factory=lambda *args:Encoder(),resolver=lambda *args:(checkpoint,'tokenizer'))
    extraction.extract_baseline(data,'clip',output,**kwargs)
    with np.load(output/'features.npz') as archive:
        assert archive['video_ids'].tolist()==['100','200','201']
        assert archive['text_video_ids'].tolist()==['200','201','200']
        assert archive['video_features'].shape==(3,512)
    before=list(calls);extraction.extract_baseline(data,'clip',output,**kwargs)
    assert calls==before
    assert not (output/'evaluation.json').exists()
    (tmp_path/'raw_videos/100.mp4').write_bytes(b'changed video')
    extraction.extract_baseline(data,'clip',output,**kwargs)
    assert len(calls)==len(before)+1


def test_composed_query_cache_uses_image_and_text_and_invalidates_changes(tmp_path):
    from visual_retrieval.composed.webvid_covr import extract_queries
    make_subset(tmp_path);data=load_subset(tmp_path)
    base,trained=tmp_path/'base.pt',tmp_path/'trained.ckpt';base.write_bytes(b'base');trained.write_bytes(b'trained')
    calls=[]
    class Encoder:
        def __init__(self,*args,**kwargs):pass
        def encode_query(self,image,text):
            calls.append((image.shape,text))
            token=np.zeros((32,256),np.float32);token[:,0 if text=='blue' else 1]=1
            return token,token[0]
    output=tmp_path/'queries'
    extract_queries(data,output,base,trained,device='cpu',factory=Encoder)
    assert len(calls)==3
    extract_queries(data,output,base,trained,device='cpu',factory=Encoder)
    assert len(calls)==3
    Image.new('RGB',(8,8),'red').save(tmp_path/'reference_frames/100.jpg')
    extract_queries(data,output,base,trained,device='cpu',factory=Encoder)
    assert len(calls)==5  # both queries using this reference
    data.queries[1]['text']='blue'
    extract_queries(data,output,base,trained,device='cpu',factory=Encoder)
    assert len(calls)==6


def unit_tokens(axis):
    x=np.zeros((15,32,256),np.float32);x[:,:,axis]=1
    return x


def evaluation_fixture(root):
    from visual_retrieval.composed.gallery import FORMAT,digest,token_path
    from visual_retrieval.composed.webvid_covr import extract_queries
    make_subset(root);data=load_subset(root)
    base,trained=root/'base.pt',root/'trained.ckpt';base.write_bytes(b'base');trained.write_bytes(b'trained')
    gallery,queries=root/'encoded/covr2',root/'encoded/covr2_queries'
    (gallery/'tokens').mkdir(parents=True)
    rows=[]
    for item in data.items:
        stat=item['path'].stat()
        rows.append(dict(item,path=str(item['path']),source_signature=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)))
        np.save(token_path(gallery,item['item_id']),unit_tokens(1 if item['item_id']=='201' else 0))
    (gallery/'manifest.json').write_text(json.dumps(dict(identity=dict(format=FORMAT,base_sha256=digest(base),frames=15,precision='fp32'),gallery_items=rows)))
    class Encoder:
        def __init__(self,*a,**k):pass
        def encode_query(self,image,text):
            token=unit_tokens(0 if text=='blue' else 1)[0]
            return token,token[0]
    extract_queries(data,queries,base,trained,device='cpu',factory=Encoder)
    return data,gallery,queries


def test_covr2_only_evaluation_full_ranks_and_labels_with_duplicate_targets(tmp_path):
    from visual_retrieval.evaluation.webvid_covr import evaluate
    data,gallery,queries=evaluation_fixture(tmp_path)
    output=tmp_path/'report'
    report=evaluate(data,gallery,queries,output,top_k=1)
    assert report['model']=='covr2' and report['gallery_size']==2 and report['num_queries']==3
    assert report['metrics']['r1']==pytest.approx(200/3)
    assert report['metrics']['mrr']==pytest.approx((1+1+.5)/3)
    records=[json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
    assert [r['rank'] for r in records]==[1,1,2]
    assert [r['ground_truth_video_id'] for r in records]==['200','201','200']
    assert all(r['reference_id'] not in [p['video_id'] for p in r['top_k']] for r in records)
    assert len(records[2]['top_k'])==1 and not records[2]['top_k'][0]['is_ground_truth']
    assert report['subset_evaluation'] and report['frame_weighting']=='runtime_modification_itc'


def test_missing_or_stale_tokens_are_not_silently_evaluated(tmp_path):
    from visual_retrieval.evaluation.webvid_covr import evaluate
    from visual_retrieval.composed.gallery import token_path
    data,gallery,queries=evaluation_fixture(tmp_path)
    token_path(gallery,'200').write_bytes(b'')
    with pytest.raises(ValueError,match='gallery|Gallery'):evaluate(data,gallery,queries,tmp_path/'report')
    report=evaluate(data,gallery,queries,tmp_path/'report',allow_incomplete=True)
    assert report['num_queries_requested']==3 and report['num_queries']==1
    assert len(report['excluded_queries'])==2 and report['excluded_gallery_ids']==['200']


def test_model_provenance_mismatch_is_rejected_even_with_allow_incomplete(tmp_path):
    from visual_retrieval.evaluation.webvid_covr import evaluate
    data,gallery,queries=evaluation_fixture(tmp_path)
    path=gallery/'manifest.json';doc=json.loads(path.read_text());doc['identity']['base_sha256']='different'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='provenance|checkpoint'):
        evaluate(data,gallery,queries,tmp_path/'report',allow_incomplete=True)


def test_pipeline_encodes_three_baselines_but_only_evaluates_covr2(tmp_path,monkeypatch):
    from visual_retrieval.cli import webvid_covr as cli
    make_subset(tmp_path)
    for name in ('base.pth','trained.ckpt','clip4clip.bin'):(tmp_path/name).write_bytes(b'weights')
    calls=[]
    monkeypatch.setattr(cli,'extract_baseline',lambda data,model,*a,**kw:calls.append('encode:'+model))
    monkeypatch.setattr(cli,'prepare_gallery',lambda *a,**kw:(calls.append('encode:covr2_gallery') or {}))
    monkeypatch.setattr(cli,'extract_queries',lambda *a,**kw:calls.append('encode:covr2_queries'))
    monkeypatch.setattr(cli,'evaluate',lambda *a,**kw:calls.append('evaluate:covr2'))
    result=cli.main(['--root',str(tmp_path),'--device','cpu','--covr2-base-checkpoint',str(tmp_path/'base.pth'),
        '--covr2-checkpoint',str(tmp_path/'trained.ckpt'),'--clip4clip-checkpoint',str(tmp_path/'clip4clip.bin')])
    assert result==0
    assert calls==['encode:clip','encode:clip4clip','encode:internvideo2','encode:covr2_gallery','encode:covr2_queries','evaluate:covr2']
    calls.clear()
    assert cli.main(['--root',str(tmp_path),'--stage','evaluate'])==0
    assert calls==['evaluate:covr2']  # no baseline checkpoints or encoders needed


def test_evaluate_uses_full_gallery_even_when_exporting_only_one_candidate(tmp_path):
    from visual_retrieval.evaluation.webvid_covr import evaluate
    from visual_retrieval.composed.gallery import token_path
    data,gallery,queries=evaluation_fixture(tmp_path)
    data.queries=data.queries[:1]
    doc=json.loads((gallery/'manifest.json').read_text())
    np.save(token_path(gallery,'200'),unit_tokens(1))
    np.save(token_path(gallery,'201'),unit_tokens(0))
    for i in range(118):
        id=str(1000+i);path=tmp_path/f'raw_videos/{id}.mp4';path.write_bytes(b'video');stat=path.stat()
        item=dict(item_id=id,source_video_id=id,path=path,start=None,end=None)
        data.items.append(item)
        doc['gallery_items'].append(dict(item,path=str(path),source_signature=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)))
        np.save(token_path(gallery,id),unit_tokens(0))
    (gallery/'manifest.json').write_text(json.dumps(doc))
    report=evaluate(data,gallery,queries,tmp_path/'report',gallery='all',top_k=1)
    prediction=json.loads((tmp_path/'report/predictions.jsonl').read_text())
    assert prediction['rank']==120 and prediction['candidate_count']==120
    assert len(prediction['top_k'])==1 and report['metrics']['r50']==0


def test_changed_query_and_changed_gallery_video_require_reencoding(tmp_path):
    from visual_retrieval.evaluation.webvid_covr import evaluate
    data,gallery,queries=evaluation_fixture(tmp_path)
    data.queries[0]['text']='new modification'
    with pytest.raises(ValueError,match='changed|extract'):
        evaluate(data,gallery,queries,tmp_path/'report')
    data.queries[0]['text']='blue'
    (tmp_path/'raw_videos/200.mp4').write_bytes(b'changed payload')
    with pytest.raises(ValueError,match='gallery'):
        evaluate(data,gallery,queries,tmp_path/'report')


def test_real_media_pipeline_and_resume_keep_only_one_encoder_alive(tmp_path,monkeypatch):
    import weakref
    from test_demo_media import source_video
    from visual_retrieval.cli import webvid_covr as cli
    from visual_retrieval.features.webvid_covr import extract_baseline
    from visual_retrieval.composed.prepare import prepare
    from visual_retrieval.composed.webvid_covr import extract_queries
    make_subset(tmp_path)
    for path in (tmp_path/'raw_videos').glob('*.mp4'):source_video(path)
    weights=tmp_path/'weights.pt';weights.write_bytes(b'fake test weights')
    alive=weakref.WeakSet();phases=[]
    class Base:
        effective_precision='fp32'
        def __init__(self,name):
            assert not alive;alive.add(self);phases.append(name)
        def encode_videos(self,clips):
            assert len(clips[0]) in (4,12)
            return np.ones((len(clips),512),np.float32)
        def encode_text(self,texts):return np.ones((len(texts),512),np.float32)
    class Gallery:
        def __init__(self,*a,**kw):
            assert not alive;alive.add(self);phases.append('covr2_gallery');self.count=0
        def encode_frames(self,frames):
            assert len(frames)==15
            self.count+=1;return unit_tokens(1 if self.count==3 else 0)
    class Query:
        def __init__(self,*a,**kw):
            assert not alive;alive.add(self);phases.append('covr2_queries')
        def encode_query(self,image,text):
            assert image.shape==(8,8,3)
            tokens=unit_tokens(0 if text=='blue' else 1)[0];return tokens,tokens[0]
    monkeypatch.setattr(cli,'extract_baseline',lambda data,model,out,**kw:extract_baseline(data,model,out,
        **kw,factory=lambda *a:Base(model),resolver=lambda *a:(weights,'fake-tokenizer')))
    monkeypatch.setattr(cli,'prepare_gallery',lambda *a,**kw:prepare(*a,**kw,encoder_factory=Gallery))
    monkeypatch.setattr(cli,'extract_queries',lambda *a,**kw:extract_queries(*a,**kw,factory=Query))
    args=['--root',str(tmp_path),'--device','cpu','--covr2-base-checkpoint',str(weights),
          '--covr2-checkpoint',str(weights),'--clip4clip-checkpoint',str(weights)]
    assert cli.main(args)==0
    assert phases==['clip','clip4clip','internvideo2','covr2_gallery','covr2_queries'] and not alive
    assert cli.main(args)==0  # every stage resumes without constructing another encoder
    assert len(phases)==5
    output=tmp_path/'features/webvid_covr/test'
    for model in ('clip','clip4clip','internvideo2'):
        assert (output/model/'features.npz').exists()
        assert not (output/model/'evaluation.json').exists()
    report=json.loads((output/'results/covr2/evaluation.json').read_text())
    assert report['num_queries']==3 and report['gallery_size']==2
    weights.unlink()
    assert cli.main(['--root',str(tmp_path),'--stage','evaluate'])==0
    assert len(phases)==5


def test_partial_download_requires_explicit_allow_incomplete(tmp_path):
    path=make_subset(tmp_path);doc=json.loads(path.read_text());doc['complete']=False;doc['requested_samples']=5
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='incomplete'):load_subset(tmp_path)
    assert len(load_subset(tmp_path,allow_incomplete=True).queries)==3


def test_reference_image_cannot_be_swapped_with_another_video(tmp_path):
    path=make_subset(tmp_path);doc=json.loads(path.read_text())
    doc['samples'][0]['reference_image']='reference_frames/201.jpg'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='Reference image|reference image'):
        load_subset(tmp_path)


def test_partial_download_is_not_reported_as_a_complete_run(tmp_path):
    from visual_retrieval.evaluation.webvid_covr import evaluate
    data,gallery,queries=evaluation_fixture(tmp_path)
    path=data.subset_path;doc=json.loads(path.read_text())
    doc.update(complete=False,requested_samples=5,failures=[dict(row_index=10,error='unavailable')])
    path.write_text(json.dumps(doc))
    data=load_subset(tmp_path,allow_incomplete=True)
    report=evaluate(data,gallery,queries,tmp_path/'report',allow_incomplete=True)
    assert report['complete'] is False and report['download_complete'] is False
    assert report['download_failed_rows']==1 and report['download_requested_samples']==5
    assert report['num_queries_requested']==3 and report['num_queries']==3
