import json
from pathlib import Path
import subprocess
import sys
import pytest
from visual_retrieval.cli.run_baselines import make_parser, commands_for
from visual_retrieval.cli.configuration import parse_configured


def test_runner_defaults_external_data_and_model_specific_outputs():
    args=make_parser().parse_args([])
    clip=commands_for(args,'msrvtt','clip');iv2=commands_for(args,'activitynet','internvideo2')
    assert clip['output']==Path(r'S:\video_retrieval') / 'msr_vtt/features/clip_vit_b32_12f_mean'
    assert iv2['output']==Path(r'S:\video_retrieval') / 'activitynet_captions/features/internvideo2_stage2_1b_all'
    assert '--device' in clip['extract'] and clip['extract'][clip['extract'].index('--device')+1]=='cuda'


def test_config_values_are_overridden_by_cli(tmp_path):
    import argparse
    path=tmp_path/'config.yaml';path.write_text('model: clip\n')
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['clip','internvideo2'],default='internvideo2')
    args=parse_configured(parser,['--config',str(path),'--model','internvideo2'])
    assert args.model=='internvideo2'


@pytest.mark.parametrize('script',['extract_features','evaluate_baselines','compare_models','search','prepare_manifests','run_baselines','evaluate_sessions'])
def test_entrypoints_work_outside_repository(tmp_path,script):
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts'/f'{script}.py'),'--help'],cwd=tmp_path,
                          capture_output=True,text=True,check=True)
    assert 'usage:' in result.stdout


@pytest.mark.parametrize('script',['extract_clip_features','evaluate_clip_retrieval'])
def test_legacy_clip_entrypoints_work_outside_repository(tmp_path,script):
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/f'{script}.py'),'--help'],cwd=tmp_path,
                   capture_output=True,text=True,check=True)
