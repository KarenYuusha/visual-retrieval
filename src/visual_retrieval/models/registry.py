"""Lazy adapter selection and reproducible model-file provenance."""
from pathlib import Path
from visual_retrieval.common import file_signature, PREPROCESSING_V2
from visual_retrieval.config import MODEL_NAMES

MODEL_INFO = {
    'clip': dict(model_id='openai/clip-vit-base-patch32', source_commit='transformers-4.40.2',
                 preprocessing='clip_b32_rgb_resize_crop224_12centers_frame_l2_mean_text77_v1', num_frames=12),
    'clip4clip': dict(model_id='CLIP4Clip-ViT-B32-meanP-2d', source_commit='508ffa3de39ba0563a03199c440ab602a72e9b6f',
                     preprocessing='clip4clip_b32_rgb_resize_crop224_12centers_frame_l2_meanP_text32_v1', num_frames=12),
    'internvideo2': dict(model_id='OpenGVLab/InternVideo2-Stage2_1B-224p-f4',
                        source_commit='3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3',
                        preprocessing=PREPROCESSING_V2, num_frames=4),
}


def checkpoint_signature(path):
    path = Path(path).resolve()
    if path.is_file():
        return file_signature(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.iterdir() if p.is_file())
    if not any(p.name in ('model.safetensors', 'pytorch_model.bin') for p in files):
        raise ValueError('CLIP directory must contain model.safetensors or pytorch_model.bin.')
    return {'path': str(path), 'files': [{**file_signature(p), 'path': p.name} for p in files]}


def resolve_model_files(name, checkpoint=None, tokenizer=None):
    if name not in MODEL_NAMES:
        raise ValueError(f'Unknown model {name}')
    if name == 'internvideo2':
        from .internvideo2_core import resolve_checkpoint
        path = resolve_checkpoint(checkpoint)
        tokenizer = tokenizer or 'google-bert/bert-large-uncased'
    elif name == 'clip4clip':
        if checkpoint is None or not Path(checkpoint).is_file():
            raise ValueError('CLIP4Clip requires --checkpoint PATH to a trained meanP/2d ViT-B/32 .bin/.pt checkpoint.')
        path = Path(checkpoint).resolve()
        tokenizer = tokenizer or 'openai/clip-vit-base-patch32'
    else:
        if checkpoint:
            path = Path(checkpoint).resolve()
            if not path.is_dir():
                raise ValueError('CLIP --checkpoint must be a local Hugging Face model directory.')
        else:
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download('openai/clip-vit-base-patch32',
                                          allow_patterns=['*.json', '*.bin', '*.safetensors', '*.txt']))
        tokenizer = tokenizer or str(path)
    return path, tokenizer


def create_encoder(name, checkpoint, device, precision, tokenizer):
    if name == 'clip':
        from .clip_encoder import ClipEncoder
        cls = ClipEncoder
    elif name == 'clip4clip':
        from .clip4clip_encoder import Clip4ClipEncoder
        cls = Clip4ClipEncoder
    elif name == 'internvideo2':
        from .internvideo2_encoder import InternVideo2Encoder
        cls = InternVideo2Encoder
    else:
        raise ValueError(f'Unknown model {name}')
    return cls(checkpoint, device, precision, tokenizer)


def validate_preprocessing(manifest):
    name = manifest.get('model_name', 'internvideo2')
    if name not in MODEL_INFO or manifest.get('preprocessing') != MODEL_INFO[name]['preprocessing']:
        raise ValueError('Feature preprocessing is incompatible or predates the tokenizer correction; rerun extraction.')
    return name
