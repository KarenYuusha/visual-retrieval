from dataclasses import replace

import numpy as np

from test_covr2_service import composed_fixture,tokens
from test_demo_service import saved_gallery,Encoder
from visual_retrieval.demo.config import DemoConfig
from visual_retrieval.demo.service import SearchService


def test_default_registry_finds_extracted_webvid_features_and_searches_numeric_ids(tmp_path):
    root=tmp_path/'webvid_covr'
    old=saved_gallery(root,ids=('100','200'),features=[[1.,0.],[0.,1.]])
    target=root/'features/webvid_covr/test/clip';target.parent.mkdir(parents=True)
    old.rename(target)
    config=DemoConfig.defaults(tmp_path)
    spec=next(d for d in config.datasets if d.key=='webvid_covr')
    assert spec.feature_dirs['clip']==target
    assert spec.covr2_features==root/'features/webvid_covr/test/covr2'
    service=SearchService(config.datasets,'cpu',lambda *a:Encoder())
    assert ('WebVid-CoVR','webvid_covr') in service.dataset_choices('clip')
    rows=service.search('alpha','clip','webvid_covr',top_k=2)
    assert [r['item_id'] for r in rows]==['100','200']
    assert rows[0]['path']==str((root/'raw_videos/100.mp4').resolve())


def test_yaml_can_select_webvid_split_and_custom_covr2_output(tmp_path):
    path=tmp_path/'demo.yaml'
    path.write_text('''datasets:
  webvid_covr:
    root: videos
    features:
      clip: features/webvid_covr/val/clip
    covr2_features: features/webvid_covr/val/covr2
''')
    spec=DemoConfig.from_yaml(path).datasets[0]
    assert spec.covr2_features==tmp_path/'videos/features/webvid_covr/val/covr2'
    assert spec.feature_dirs['clip']==tmp_path/'videos/features/webvid_covr/val/clip'


def test_webvid_extractor_output_loads_in_demo_without_conversion(tmp_path,monkeypatch):
    from test_webvid_covr_evaluation import make_subset
    from visual_retrieval.data.webvid_covr import load_subset
    from visual_retrieval.features import webvid_covr as extraction
    root=tmp_path/'webvid_covr';root.mkdir()
    make_subset(root);data=load_subset(root)
    weights=tmp_path/'weights';weights.write_bytes(b'weights')
    class Model:
        effective_precision='fp32'
        def encode_videos(self,clips):return np.ones((len(clips),512),np.float32)
        def encode_text(self,texts):return np.ones((len(texts),512),np.float32)
    monkeypatch.setattr(extraction,'read_video_frames',lambda path,count_frames:
        (np.zeros((count_frames,2,2,3),np.uint8),list(range(count_frames)),count_frames))
    extraction.extract_baseline(data,'clip',root/'features/webvid_covr/test/clip',device='cpu',
        factory=lambda *a:Model(),resolver=lambda *a:(weights,'tokenizer'))
    service=SearchService(DemoConfig.defaults(tmp_path).datasets,'cpu',lambda *a:Model())
    assert service.gallery_counts('clip')=={'webvid_covr':3}
    rows=service.search('a person','clip','webvid_covr',top_k=3)
    assert [r['item_id'] for r in rows]==['100','200','201']


def test_composed_search_uses_configured_tokens_and_keeps_raw_video_root(tmp_path):
    from visual_retrieval.demo.composed import ComposedSearchService
    class Query:
        def __init__(self,*a,**kw):pass
        def encode_query(self,*a):return tokens()[0],np.eye(256,dtype=np.float32)[0]
    old,spec,folder=composed_fixture(tmp_path,Query)
    custom=tmp_path/'custom_outputs/covr2';custom.parent.mkdir()
    folder.rename(custom)
    configured=replace(spec,covr2_features=custom)
    service=ComposedSearchService(old.text,[configured],old.base,old.checkpoint,Query)
    assert service.composed_available
    result=service.search_composed(np.zeros((4,4,3),np.uint8),'run',dataset=spec.key)[0]
    assert result['path']==str((spec.root/'raw_videos/same.mp4').resolve())
    assert service.search_composed(np.zeros((4,4,3),np.uint8),'run',source=(spec.key,'same'))==[]


def test_demo_preflight_reports_actual_combined_precision_mismatch(tmp_path,capsys):
    import json
    from visual_retrieval.demo.app import main
    first=saved_gallery(tmp_path/'msr_vtt')
    first.rename(first.with_name('clip_vit_b32_12f_mean'))
    second=saved_gallery(tmp_path/'webvid_covr')
    path=second/'manifest.json';doc=json.loads(path.read_text())
    doc['effective_precision']='fp16';path.write_text(json.dumps(doc))
    output=tmp_path/'webvid_covr/features/webvid_covr/test/clip'
    output.parent.mkdir(parents=True);second.rename(output)
    main(['--base-root',str(tmp_path),'--check','--device','cpu'])
    text=capsys.readouterr().out
    assert 'Combined clip mismatch: MSR-VTT vs WebVid-CoVR: precision (fp32 vs fp16)' in text
