"""Map a trained HF CLIP4Clip encoder to the original meanP CLIP layout.

Inverse of Transformers' CLIP conversion: split HF Q/K/V -> packed attention,
and transpose linear projection weights. No base-model weights are substituted.
Reference: transformers v4.40.2 models/clip/convert_clip_original_pytorch_to_hf.py
"""
import re
import torch


def convert_hf_clip_weights(state):
    used = set()
    def take(key):
        if key not in state or not isinstance(state[key], torch.Tensor):
            raise ValueError(f'Missing trained HF CLIP4Clip tensor: {key}')
        used.add(key)
        return state[key]

    direct = {
        'logit_scale': 'logit_scale',
        'token_embedding.weight': 'text_model.embeddings.token_embedding.weight',
        'positional_embedding': 'text_model.embeddings.position_embedding.weight',
        'ln_final.weight': 'text_model.final_layer_norm.weight',
        'ln_final.bias': 'text_model.final_layer_norm.bias',
        'visual.conv1.weight': 'vision_model.embeddings.patch_embedding.weight',
        'visual.class_embedding': 'vision_model.embeddings.class_embedding',
        'visual.positional_embedding': 'vision_model.embeddings.position_embedding.weight',
        'visual.ln_pre.weight': 'vision_model.pre_layrnorm.weight',
        'visual.ln_pre.bias': 'vision_model.pre_layrnorm.bias',
        'visual.ln_post.weight': 'vision_model.post_layernorm.weight',
        'visual.ln_post.bias': 'vision_model.post_layernorm.bias',
    }
    result = {native: take(hf) for native, hf in direct.items()}
    result['visual.proj'] = take('visual_projection.weight').T.contiguous()
    result['text_projection'] = take('text_projection.weight').T.contiguous()

    for hf_prefix, native_prefix in [('vision_model', 'visual.transformer'), ('text_model', 'transformer')]:
        pattern = re.compile(rf'^{hf_prefix}\.encoder\.layers\.(\d+)\.')
        layers = {int(match.group(1)) for key in state if (match := pattern.match(key))}
        if not layers or layers != set(range(max(layers) + 1)):
            raise ValueError(f'Missing or noncontiguous HF CLIP4Clip layers: {hf_prefix}')
        for layer in sorted(layers):
            hf = f'{hf_prefix}.encoder.layers.{layer}.'
            native = f'{native_prefix}.resblocks.{layer}.'
            for suffix in ('weight', 'bias'):
                result[native + f'attn.in_proj_{suffix}'] = torch.cat(
                    [take(hf + f'self_attn.{part}_proj.{suffix}') for part in ('q', 'k', 'v')], dim=0)
                for original, converted in [('attn.out_proj', 'self_attn.out_proj'),
                                            ('ln_1', 'layer_norm1'), ('ln_2', 'layer_norm2'),
                                            ('mlp.c_fc', 'mlp.fc1'), ('mlp.c_proj', 'mlp.fc2')]:
                    result[native + f'{original}.{suffix}'] = take(hf + f'{converted}.{suffix}')

    # Older Transformers versions persisted these deterministic integer buffers.
    for prefix, embedding in [('text_model', 'positional_embedding'), ('vision_model', 'visual.positional_embedding')]:
        key = f'{prefix}.embeddings.position_ids'
        if key in state:
            ids = take(key)
            expected = torch.arange(result[embedding].shape[0], device=ids.device).reshape(1, -1)
            if ids.dtype not in (torch.int32, torch.int64) or not torch.equal(ids, expected):
                raise ValueError(f'Invalid HF position_ids buffer: {key}')
    unexpected = set(state) - used
    if unexpected:
        raise ValueError(f'Unexpected HF CLIP4Clip weights: {sorted(unexpected)}')
    return result
