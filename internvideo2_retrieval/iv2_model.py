"""Minimal inference wrapper around the official InternVideo2 Stage2 1B code."""
import gc
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from transformers import BertTokenizer

from iv2_vendor.internvideo2 import pretrain_internvideo2_1b_patch14_224
from iv2_vendor.xbert import BertConfig, BertForMaskedLM
from retrieval_common import clean_text, middle_indices

MODEL_ID = 'OpenGVLab/InternVideo2-Stage2_1B-224p-f4'
CHECKPOINT_NAME = 'InternVideo2-stage2_1b-224p-f4.pt'
SOURCE_COMMIT = '3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3'


class InternVideo2Tokenizer(BertTokenizer):
    """Match the official tokenizer's single-caption [CLS] + text format.

    The standard Hugging Face BERT tokenizer appends [SEP]. InternVideo2's
    models/backbones/bert/tokenization_bert.py deliberately omits it for single
    sequences. Retain the maintained tokenizer implementation while restoring
    that model-specific behavior and consistent masks/token-type lengths.
    """

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0
        return super().build_inputs_with_special_tokens(token_ids_0, token_ids_1)

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if token_ids_1 is None and not already_has_special_tokens:
            return [1] + [0] * len(token_ids_0)
        return super().get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens)

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 1)
        return super().create_token_type_ids_from_sequences(token_ids_0, token_ids_1)


def resolve_checkpoint(explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint is not a file: {path}')
        return path
    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(MODEL_ID, CHECKPOINT_NAME))
    except Exception as error:
        raise RuntimeError(f'Cannot download {MODEL_ID}. If access is gated, accept its '
                           'conditions on Hugging Face and run huggingface-cli login. '
                           'Or supply --checkpoint PATH_TO_PT_FILE.') from error


def canonical_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must contain a state dictionary.')
    for key in ('model', 'module', 'state_dict'):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    state = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        while key.startswith('module.'):
            key = key[len('module.'):]
        if key in state:
            raise ValueError(f'Duplicate checkpoint key after unwrapping: {key}')
        state[key] = value
    if not state:
        raise ValueError('No tensor weights found in checkpoint.')
    return state


def validate_retrieval_weights(missing_keys, fusion_layer=19):
    required_missing = []
    for key in missing_keys:
        # This deterministic buffer is reconstructed by BERT, and official
        # BertPreTrainedModel also excludes it from missing-weight warnings.
        if key == 'text_encoder.bert.embeddings.position_ids':
            continue
        unused = key.startswith(('vision_encoder.clip_decoder.',
                                 'vision_encoder.final_clip_decoder.',
                                 'vision_encoder.clip_pos_embed',
                                 'vision_encoder.clip_img_pos_embed',
                                 'vision_encoder.img_pos_embed', 'text_encoder.cls.'))
        layer = re.match(r'text_encoder\.bert\.encoder\.layer\.(\d+)\.', key)
        if layer and int(layer.group(1)) >= fusion_layer:
            unused = True
        if not unused:
            required_missing.append(key)
    if required_missing:
        raise ValueError('Checkpoint is missing weights used for retrieval (do not continue '
                         'with random projections/backbones): ' + ', '.join(required_missing[:20]))


class RetrievalModel(nn.Module):
    def __init__(self):
        super().__init__()
        from easydict import EasyDict
        config = EasyDict(vision_encoder=dict(
            clip_embed_dim=768, num_frames=4, tubelet_size=1,
            sep_image_video_pos_embed=True, use_checkpoint=False, checkpoint_num=0,
            clip_teacher_embed_dim=3200, clip_teacher_final_dim=768,
            clip_norm_type='l2', clip_return_layer=6, clip_student_return_interval=1,
            pretrained=None, use_flash_attn=False, use_fused_rmsnorm=False, use_fused_mlp=False,
        ))
        self.vision_encoder = pretrain_internvideo2_1b_patch14_224(config)
        bert_config = BertConfig.from_json_file(
            str(Path(__file__).parent / 'iv2_vendor' / 'config_bert_large.json'))
        bert_config.encoder_width = 1408
        bert_config.fusion_layer = 19
        bert_config.gradient_checkpointing = False
        # All trained BERT weights come from the full Stage2 checkpoint.
        self.text_encoder = BertForMaskedLM(bert_config)
        self.vision_proj = nn.Linear(768, 512)
        self.text_proj = nn.Linear(1024, 512)

    @torch.inference_mode()
    def video_features(self, frames):
        dtype = self.vision_encoder.patch_embed.proj.weight.dtype
        video = frames.permute(0, 2, 1, 3, 4).to(dtype=dtype)
        # Exactly the same backbone + pooling used by official get_vid_feat;
        # do not execute training-only distillation decoders.
        tokens = self.vision_encoder(video, mask=None, use_image=False, x_vis_only=True)
        pooled = self.vision_encoder.clip_projector(tokens)
        return F.normalize(self.vision_proj(pooled).float(), dim=-1)

    @torch.inference_mode()
    def text_features(self, tokens):
        output = self.text_encoder.bert(tokens['input_ids'],
                                        attention_mask=tokens['attention_mask'],
                                        return_dict=True, mode='text')
        return F.normalize(self.text_proj(output.last_hidden_state[:, 0]).float(), dim=-1)


def load_model(checkpoint, device='cuda', precision='auto', tokenizer_path='google-bert/bert-large-uncased'):
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install CUDA-enabled PyTorch, or use --device cpu '
                           '(very slow for the 1B model).')
    if precision == 'auto':
        precision = ('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if device.type == 'cuda' else 'fp32'
    if device.type == 'cpu' and precision != 'fp32':
        raise ValueError('Use fp32 precision on CPU.')
    if precision == 'bf16' and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bf16; use --precision fp16.')
    dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[precision]
    print(f'Loading full Stage2 checkpoint: {checkpoint}; device={device}; precision={precision}')
    # Initialize directly in the inference dtype to reduce host memory pressure.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model = RetrievalModel()
    finally:
        torch.set_default_dtype(previous_dtype)
    state = canonical_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    result = model.load_state_dict(state, strict=False)
    validate_retrieval_weights(result.missing_keys)
    del state
    gc.collect()
    model = model.eval().requires_grad_(False).to(device=device, dtype=dtype)
    tokenizer = InternVideo2Tokenizer.from_pretrained(tokenizer_path)
    print(f'Validated all retrieval weights. Ignored {len(result.unexpected_keys)} extra checkpoint keys.')
    return model, tokenizer, precision


def read_video_frames(path, start=None, end=None):
    # Count actually decodable frames with a first pass. Do not trust container metadata.
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    try:
        timestamps = []
        count = 0
        while cap.grab():
            if start is not None or end is not None:
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            count += 1
    finally:
        cap.release()
    if start is None and end is None:
        indices = middle_indices(count, 4)
    elif start is not None and end is not None:
        from dataset_adapters import segment_indices
        indices = segment_indices(timestamps, start, end)
    else:
        raise ValueError("Both start and end are required for a segment.")
    wanted = set(indices)
    cap = cv2.VideoCapture(str(path))
    frames = {}
    try:
        for index in range(indices[-1] + 1):
            if not cap.grab():
                raise ValueError(f'Video failed while sampling frame {index}: {path}')
            if index in wanted:
                ok, bgr = cap.retrieve()
                if not ok or bgr is None:
                    raise ValueError(f'Cannot decode frame {index}: {path}')
                frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return np.stack([frames[index] for index in indices]), indices, count


def prepare_frames(rgb_frames):
    frames = torch.from_numpy(np.ascontiguousarray(rgb_frames)).permute(0, 3, 1, 2)
    # Matches official get_test_transform: uint8 bicubic resize, then float/255,
    # ImageNet mean/std. Input layout is [1, T, C, H, W].
    frames = TF.resize(frames, [224, 224], interpolation=InterpolationMode.BICUBIC, antialias=True)
    frames = TF.normalize(frames.float().div(255), [.485, .456, .406], [.229, .224, .225])
    return frames.unsqueeze(0)


def encode_text_batch(model, tokenizer, captions, device):
    tokens = tokenizer([clean_text(text) for text in captions], padding='max_length',
                       truncation=True, max_length=40, return_tensors='pt').to(device)
    return model.text_features(tokens).cpu().numpy().astype(np.float32)
