"""External runtime paths. Repository data/ is never selected implicitly."""
from pathlib import Path
DEFAULT_BASE = Path(r'S:\video_retrieval')
DATASET_NAMES = ('msrvtt', 'vatex', 'activitynet')
DATASET_ALIASES = {'activitynet_captions': 'activitynet'}
MODEL_NAMES = ('clip', 'clip4clip', 'internvideo2')
MODEL_OUTPUTS = {'clip': 'clip_vit_b32_12f_mean', 'clip4clip': 'clip4clip_vit_b32_12f_meanP',
                 'internvideo2': 'internvideo2_stage2_1b_all'}

def canonical_dataset(name):
    name = DATASET_ALIASES.get(name, name)
    if name not in DATASET_NAMES:
        raise ValueError(f'Unknown dataset: {name}')
    return name

def dataset_root(name, base=DEFAULT_BASE):
    return Path(base) / canonical_dataset(name)

def output_directory(model, root):
    return Path(root) / 'features' / MODEL_OUTPUTS[model]
