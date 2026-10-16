"""Minimal, weight-checked CoVR-2 / BLIP-2 inference (no language decoder)."""
from functools import partial
import re

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


# This checkpoint contains a scheduler object. This inert replacement accepts its
# serialized attributes without importing or running upstream training code.
class CosineSchedule:
    pass


CosineSchedule.__module__ = 'src.tools.scheduler'


def read_state(path):
    with torch.serialization.safe_globals([CosineSchedule, set]):
        data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    state = data.get('model', data.get('state_dict', data))
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Checkpoint must contain a tensor model state.')
    return state


def load_required(module, state, prefix):
    expected = module.state_dict()
    missing = [k for k in expected if prefix + k not in state]
    if missing:
        raise ValueError(f'Checkpoint missing required {prefix} tensors: {missing[:5]}')
    selected = {k: state[prefix + k] for k in expected}
    for k, tensor in selected.items():
        if tensor.shape != expected[k].shape:
            raise ValueError(f'Checkpoint shape mismatch: {prefix}{k}')
    module.load_state_dict(selected, strict=True, assign=True)
    return module


def composed_tokens(qformer, projection, queries, image, ids, mask):
    queries = queries.expand(image.shape[0], -1, -1)
    query_mask = torch.ones(queries.shape[:2], dtype=mask.dtype, device=image.device)
    out = qformer.bert(ids, query_embeds=queries,
        attention_mask=torch.cat([query_mask, mask], dim=1),
        encoder_hidden_states=image,
        encoder_attention_mask=torch.ones(image.shape[:-1], dtype=torch.long, device=image.device),
        return_dict=True)
    return F.normalize(projection(out.last_hidden_state[:, :queries.shape[1]]), dim=-1)


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).to(x.dtype)


def _qformer(state):
    from .covr2_vendor.Qformer import BertConfig, BertLMHeadModel
    config = BertConfig(vocab_size=30523)
    config.encoder_width, config.add_cross_attention = 1408, True
    config.cross_attention_freq, config.query_length = 2, 32
    with torch.device('meta'):
        model = BertLMHeadModel(config)
        proj = nn.Linear(768, 256)
    load_required(model, state, 'Qformer.')
    load_required(proj, state, 'text_proj.')
    queries = state.get('query_tokens')
    if queries is None or queries.shape != (1, 32, 768):
        raise ValueError('Checkpoint missing valid query_tokens.')
    return model, proj, queries


def caption(text):
    return re.sub(r'\s+', ' ', re.sub(r'([.!"()*#:;~])', ' ', text.lower()).replace('-', ' ').replace('/', ' ')).strip()


class CoVR2Encoder:
    """Gallery mode loads COCO weights; query mode loads trained composed weights.

    The original frozen COCO Q-Former is retained for text-dependent frame
    weighting. Only one vision encoder is resident in either mode.
    """
    def __init__(self, base_checkpoint, query_checkpoint=None, device='cuda', tokenizer='bert-base-uncased'):
        from transformers import BertTokenizer
        from .covr2_vendor.eva_vit import VisionTransformer
        self.gallery_mode = query_checkpoint is None
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        self.effective_precision = 'fp16' if self.dtype == torch.float16 else 'fp32'
        base = read_state(base_checkpoint)
        state = read_state(query_checkpoint) if query_checkpoint else base
        with torch.device('meta'):
            self.visual = VisionTransformer(img_size=364, patch_size=14, embed_dim=1408,
                depth=39, num_heads=16, mlp_ratio=4.3637, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), use_checkpoint=False)
            self.norm = LayerNorm(1408)
            self.vision_proj = nn.Linear(768, 256)
        load_required(self.visual, state, 'visual_encoder.')
        load_required(self.norm, state, 'ln_vision.')
        load_required(self.vision_proj, base, 'vision_proj.')
        self.qformer, self.text_proj, self.queries = _qformer(state)
        if query_checkpoint:
            self.frozen_qformer, self.frozen_text_proj, _ = _qformer(base)
        else:
            self.frozen_qformer, self.frozen_text_proj = self.qformer, self.text_proj
        for module in {self.visual, self.norm, self.vision_proj, self.qformer, self.text_proj,
                       self.frozen_qformer, self.frozen_text_proj}:
            module.eval().requires_grad_(False).to(device=self.device, dtype=self.dtype)
        self.queries = self.queries.to(device=self.device, dtype=self.dtype)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer, truncation_side='right')
        self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})

    def _image(self, images):
        arrays = [np.asarray(Image.fromarray(np.asarray(im, dtype=np.uint8)).convert('RGB').resize(
            (364, 364), Image.Resampling.BICUBIC), dtype=np.float32) / 255 for im in images]
        x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
        mean = torch.tensor([.48145466, .4578275, .40821073]).view(1,3,1,1)
        std = torch.tensor([.26862954, .26130258, .27577711]).view(1,3,1,1)
        return self.norm(self.visual(((x - mean) / std).to(self.device, dtype=self.dtype)))

    def _text(self, text):
        return self.tokenizer([caption(text)], padding='longest', truncation=True,
                              max_length=64, return_tensors='pt').to(self.device)

    @torch.inference_mode()
    def encode_frames(self, frames):
        # Batch one keeps EVA's attention footprint suitable for a 12 GB GPU.
        if not self.gallery_mode:
            raise ValueError('Gallery extraction requires the frozen COCO model, not composed query weights.')
        result = []
        for frame in frames:
            image = self._image([frame])
            out = self.qformer.bert(query_embeds=self.queries,
                encoder_hidden_states=image,
                encoder_attention_mask=torch.ones(image.shape[:-1], dtype=torch.long, device=self.device),
                return_dict=True).last_hidden_state
            result.append(F.normalize(self.vision_proj(out), dim=-1)[0].float().cpu().numpy())
        return np.stack(result)

    @torch.inference_mode()
    def encode_query(self, image, text):
        tokens = self._text(text)
        result = composed_tokens(self.qformer, self.text_proj, self.queries,
                                self._image([image]), tokens.input_ids, tokens.attention_mask)
        frozen = self.frozen_qformer.bert(tokens.input_ids, attention_mask=tokens.attention_mask,
                                         return_dict=True).last_hidden_state[:, 0]
        scoring = F.normalize(self.frozen_text_proj(frozen), dim=-1)
        return result[0].float().cpu().numpy(), scoring[0].float().cpu().numpy()
