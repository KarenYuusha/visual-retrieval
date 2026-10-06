import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from visual_retrieval.common import file_signature, sha256, write_json
from visual_retrieval.features.cache import atomic_npz
from visual_retrieval.models.registry import MODEL_INFO, checkpoint_signature


@pytest.mark.parametrize('model',['clip','clip4clip'])
def test_cached_model_pipeline_does_not_load_weights(tmp_path,model):
    root=tmp_path/'dataset';root.mkdir();(root/'raw_videos').mkdir()
    write_json(root/'subset.json',dict(train=['video1'],test=['video2']))
    write_json(root/'MSRVTT_data.json',dict(sentences=[dict(video_id='video1',caption='one'),dict(video_id='video2',caption='two')]))
    for video in ('video1','video2'):
        (root/'raw_videos'/f'{video}.mp4').write_bytes(b'not decoded')
    checkpoint=tmp_path/'weights'
    if model=='clip':
        checkpoint.mkdir();(checkpoint/'pytorch_model.bin').write_bytes(b'not loaded')
        tokenizer=str(checkpoint.resolve())
    else:
        checkpoint.write_bytes(b'not loaded')
        tokenizer='openai/clip-vit-base-patch32'
    output=tmp_path/'features';cache=output/'video_cache';cache.mkdir(parents=True)
    features=np.eye(512,dtype=np.float32)[:2]
    for video,feature in zip(('video1','video2'),features):
        atomic_npz(cache/f'{video}.npz',feature=feature,source_signature=json.dumps(file_signature(root/'raw_videos'/f'{video}.mp4')))
    atomic_npz(output/'text_cache.npz',text_features=features,text_video_ids=np.array(['video1','video2']))
    info=MODEL_INFO[model]
    write_json(output/'run_config.json',dict(model_id=info['model_id'],source_commit=info['source_commit'],
               protocol='combined_train_test_all_captions',subset_sha256=sha256(root/'subset.json'),
               annotations_sha256=sha256(root/'MSRVTT_data.json'),checkpoint=checkpoint_signature(checkpoint),
               tokenizer=tokenizer,precision='auto',device='cpu',preprocessing=info['preprocessing'],model_name=model))
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    subprocess.run([sys.executable,str(scripts/'extract_features.py'),'--model',model,'--data-root',str(root),
                    '--output-dir',str(output),'--checkpoint',str(checkpoint),'--device','cpu'],check=True,cwd=tmp_path)
    subprocess.run([sys.executable,str(scripts/'evaluate_baselines.py'),'--features-dir',str(output)],check=True,cwd=tmp_path)
    report=json.loads((output/'evaluation.json').read_text())
    assert report['model_name']==model
    assert report['text_to_video']==dict(r1=100.,r5=100.,r10=100.,mrr=1.)
