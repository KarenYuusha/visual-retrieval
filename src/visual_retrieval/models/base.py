"""Encoder contract and shared inference precision/pooling."""
from abc import ABC, abstractmethod
from contextlib import nullcontext
import torch
from torch.nn import functional as F


class VideoTextEncoder(ABC):
    embedding_dim = 512
    @abstractmethod
    def encode_video(self, rgb_frames):
        """RGB uint8 [T,H,W,3] -> normalized float32 numpy [1,D]."""
    @abstractmethod
    def encode_text(self, texts):
        """List[str] -> normalized float32 numpy [B,D]."""


def resolve_precision(device, precision):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: install CUDA-enabled PyTorch or explicitly use --device cpu.')
    if precision == 'auto':
        precision = ('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if device == 'cuda' else 'fp32'
    if device == 'cpu' and precision != 'fp32':
        raise ValueError('CPU inference requires fp32.')
    if precision == 'bf16' and device == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support bf16; use fp16.')
    if precision not in ('fp32', 'fp16', 'bf16'):
        raise ValueError(f'Invalid precision: {precision}')
    return precision


def autocast_context(device, precision):
    if device == 'cuda' and precision != 'fp32':
        return torch.autocast('cuda', dtype={'fp16': torch.float16, 'bf16': torch.bfloat16}[precision])
    return nullcontext()


def pool_frame_features(features):
    """Normalize each frame, mean pool, then normalize the video."""
    return F.normalize(F.normalize(features.float(), dim=-1).mean(dim=0, keepdim=True), dim=-1)
