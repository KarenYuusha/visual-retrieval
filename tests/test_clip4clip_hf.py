import pytest
import torch
from transformers import CLIPConfig, CLIPModel

from visual_retrieval.models.clip4clip_encoder import build_trained_clip, extract_clip_weights


def small_hf_model():
    return CLIPModel(CLIPConfig(projection_dim=16,
        text_config=dict(vocab_size=20, hidden_size=64, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=1,
                         max_position_embeddings=8, bos_token_id=18, eos_token_id=19),
        vision_config=dict(hidden_size=64, intermediate_size=256, num_hidden_layers=2,
                           num_attention_heads=1, image_size=28, patch_size=14))).eval()


def test_hf_checkpoint_preserves_image_and_text_embeddings():
    torch.manual_seed(9)
    reference = small_hf_model()
    loaded = build_trained_clip(reference.state_dict(), require_vit_b32=False)
    pixels = torch.randn(3, 3, 28, 28)
    # EOT is highest token, and padding is only after EOT (same native protocol).
    tokens = torch.tensor([[18, 3, 6, 19, 0, 0, 0, 0], [18, 9, 2, 4, 19, 0, 0, 0]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded.encode_image(pixels), reference.get_image_features(pixels), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(loaded.encode_text(tokens), reference.get_text_features(tokens), rtol=1e-4, atol=1e-5)
    # Check transposes and Q/K/V order, including exact preservation of fp32 weights.
    torch.testing.assert_close(loaded.visual.proj, reference.visual_projection.weight.T, rtol=0, atol=0)
    torch.testing.assert_close(loaded.transformer.resblocks[0].attn.in_proj_weight,
        torch.cat([reference.text_model.encoder.layers[0].self_attn.q_proj.weight,
                   reference.text_model.encoder.layers[0].self_attn.k_proj.weight,
                   reference.text_model.encoder.layers[0].self_attn.v_proj.weight]), rtol=0, atol=0)


def test_hf_checkpoint_missing_trained_weight_is_rejected():
    state = dict(small_hf_model().state_dict())
    del state['vision_model.encoder.layers.1.self_attn.k_proj.weight']
    with pytest.raises(ValueError, match='Missing.*k_proj'):
        build_trained_clip(state, require_vit_b32=False)


def test_hf_checkpoint_with_temporal_head_is_rejected():
    state = dict(small_hf_model().state_dict())
    state['frame_position_embeddings.weight'] = torch.ones(1)
    with pytest.raises(ValueError, match='meanP'):
        extract_clip_weights(state)


def test_hf_checkpoint_unrecognized_weights_are_not_silently_discarded():
    state = dict(small_hf_model().state_dict())
    state['unexpected_trained_head.weight'] = torch.ones(1)
    with pytest.raises(ValueError, match='Unexpected.*unexpected_trained_head'):
        build_trained_clip(state, require_vit_b32=False)


def test_hf_checkpoint_old_position_ids_are_validated():
    state = dict(small_hf_model().state_dict())
    state['text_model.embeddings.position_ids'] = torch.arange(8).reshape(1, -1)
    state['vision_model.embeddings.position_ids'] = torch.arange(5).reshape(1, -1)
    assert extract_clip_weights(state)['text_projection'].shape == (64, 16)
    state['text_model.embeddings.position_ids'][0, 0] = 4
    with pytest.raises(ValueError, match='position_ids'):
        extract_clip_weights(state)
