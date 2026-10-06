"""Trained CLIP4Clip ViT-B/32 meanP, 2d patch inference; no fallback weights."""
import torch
from .checkpoint import load_checkpoint
from PIL import Image
from torch.nn import functional as F
from transformers import CLIPTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from .base import VideoTextEncoder, resolve_precision, autocast_context, pool_frame_features
from .clip4clip_vendor.module_clip import build_model

MODEL_ID = 'CLIP4Clip-ViT-B32-meanP-2d'
SOURCE_COMMIT = '508ffa3de39ba0563a03199c440ab602a72e9b6f'
PREPROCESSING = 'clip4clip_b32_rgb_resize_crop224_12centers_frame_l2_meanP_text32_v1'


def extract_clip_weights(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError('CLIP4Clip checkpoint must contain a state dictionary.')
    for key in ('model', 'module', 'state_dict'):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    state = {}
    for key, value in checkpoint.items():
        while key.startswith('module.'):
            key = key[7:]
        if key in state:
            raise ValueError(f'Duplicate checkpoint key: {key}')
        state[key] = value
    forbidden = ('frame_position_embeddings.', 'transformerClip.', 'lstm_visual.',
                 'cross.', 'similarity_dense.', 'clip.visual.conv2.')
    if any(key.startswith(forbidden) for key in state):
        raise ValueError('Only meanP with 2d patches is supported; checkpoint contains temporal/cross/3d weights.')
    clip = {key[5:]: value for key, value in state.items() if key.startswith('clip.') and isinstance(value, torch.Tensor)}
    if not clip:
        raise ValueError('Expected trained CLIP4Clip clip-prefixed weights; ordinary CLIP weights are not accepted.')
    return clip


def build_trained_clip(checkpoint, require_vit_b32=True):
    state = extract_clip_weights(checkpoint)
    if require_vit_b32:
        required = {'visual.conv1.weight': (768, 3, 32, 32), 'visual.proj': (768, 512),
                    'text_projection': (512, 512), 'token_embedding.weight': (49408, 512),
                    'positional_embedding': (77, 512)}
        for key, shape in required.items():
            if key not in state or tuple(state[key].shape) != shape:
                raise ValueError(f'Expected ViT-B/32 checkpoint: {key} must have shape {shape}.')
    if require_vit_b32:
        for prefix in ('visual.transformer.resblocks.', 'transformer.resblocks.'):
            layers = {int(key[len(prefix):].split('.')[0]) for key in state if key.startswith(prefix)}
            if layers != set(range(12)):
                raise ValueError('ViT-B/32 requires all 12 vision layers and all 12 text layers.')
    # Official builder loads strictly: never fill missing encoder/projection weights.
    return build_model(dict(state)).float().eval().requires_grad_(False)


class Clip4ClipEncoder(VideoTextEncoder):
    def __init__(self, checkpoint, device='cuda', precision='auto', tokenizer='openai/clip-vit-base-patch32'):
        self.device = device
        self.effective_precision = resolve_precision(device, precision)
        state = load_checkpoint(checkpoint)
        self.model = build_trained_clip(state).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer))
        self.transform = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224),
                                  ToTensor(), Normalize((.48145466, .4578275, .40821073),
                                                        (.26862954, .26130258, .27577711))])

    @torch.inference_mode()
    def encode_video(self, rgb_frames):
        frames = torch.stack([self.transform(Image.fromarray(frame)) for frame in rgb_frames]).to(self.device)
        with autocast_context(self.device, self.effective_precision):
            features = self.model.encode_image(frames, video_frame=len(frames))
        return pool_frame_features(features).cpu().numpy()

    @torch.inference_mode()
    def encode_text(self, texts):
        tokens = self.tokenizer(texts, max_length=32, truncation=True, padding='max_length', return_tensors='pt')
        # Official dataloaders pad with ID0; causal attention makes padded suffixes irrelevant.
        ids = tokens['input_ids'].masked_fill(tokens['attention_mask'] == 0, 0).to(self.device)
        with autocast_context(self.device, self.effective_precision):
            features = self.model.encode_text(ids)
        return F.normalize(features.float(), dim=-1).cpu().numpy()
