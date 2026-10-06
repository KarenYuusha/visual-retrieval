import subprocess
import sys
from pathlib import Path
from visual_retrieval.config import DEFAULT_BASE, dataset_root, output_directory


def test_external_roots_are_user_requested_names():
    assert str(DEFAULT_BASE) == r'S:\video_retrieval'
    assert dataset_root('msrvtt') == DEFAULT_BASE / 'msrvtt'
    assert dataset_root('vatex') == DEFAULT_BASE / 'vatex'
    assert dataset_root('activitynet') == DEFAULT_BASE / 'activitynet'
    assert dataset_root('activitynet_captions') == DEFAULT_BASE / 'activitynet'
    assert output_directory('clip', Path('/videos')) == Path('/videos/features/clip_vit_b32_12f_mean')


def test_lightweight_package_does_not_import_models():
    code = 'import visual_retrieval.data.adapters; import sys; assert "torch" not in sys.modules'
    root = Path(__file__).resolve().parents[1]
    import os
    env = {**os.environ, 'PYTHONPATH': str(root / 'src')}
    subprocess.run([sys.executable, '-c', code], env=env, check=True)


def test_feature_source_is_not_ignored_by_git():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(['git','check-ignore','--no-index','src/visual_retrieval/features/extraction.py'],
                          cwd=root,capture_output=True,text=True)
    assert result.returncode==1, result.stdout
