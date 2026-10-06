import numpy as np
import pytest
import torch

from iv2_model import prepare_frames, canonical_state_dict, validate_retrieval_weights, read_video_frames, RetrievalModel
from iv2_vendor.internvideo2 import Attention, RMSNorm, PretrainInternVideo2
from iv2_vendor.xbert import BertConfig, BertForMaskedLM


def test_sdpa_matches_official_attention_with_qk_normalization():
    torch.manual_seed(9)
    model = Attention(32, num_heads=4, qkv_bias=False, qk_normalization=True,
                      norm_layer=RMSNorm, use_flash_attn=False).eval()
    x = torch.randn(2, 13, 32)
    q, k, v = model.qkv(x).reshape(2, 13, 3, 4, 8).permute(2, 0, 3, 1, 4).unbind(0)
    q = model.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(2, 13, 4, 8).transpose(1, 2)
    k = model.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(2, 13, 4, 8).transpose(1, 2)
    expected = (((q * model.scale) @ k.transpose(-2, -1)).softmax(-1) @ v)
    expected = model.proj(expected.transpose(1, 2).reshape(2, 13, 32))
    torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-6)


def test_rgb_and_imagenet_normalization():
    frames = np.zeros((4, 12, 16, 3), dtype=np.uint8)
    frames[:, :, :, 0] = 255  # RGB red, not BGR
    x = prepare_frames(frames)
    assert x.shape == (1, 4, 3, 224, 224)
    torch.testing.assert_close(x[0, 0, :, 0, 0],
                              torch.tensor([(1-.485)/.229, -.456/.224, -.406/.225]))


def test_checkpoint_wrapper_and_ddp_prefix_are_unwrapped():
    weight = torch.ones(2)
    result = canonical_state_dict({'module': {'module.vision_proj.weight': weight}})
    assert list(result) == ['vision_proj.weight']
    assert result['vision_proj.weight'] is weight


def test_missing_trained_projection_cannot_pass_silently():
    with pytest.raises(ValueError, match='vision_proj.weight'):
        validate_retrieval_weights(['vision_proj.weight'], fusion_layer=19)
    # Training-only text fusion/MLM and vision distillation are not used in retrieval.
    validate_retrieval_weights(['text_encoder.cls.predictions.bias',
                                'text_encoder.bert.encoder.layer.19.output.dense.weight',
                                'vision_encoder.clip_decoder.0.head.weight',
                                'text_encoder.bert.embeddings.position_ids'], fusion_layer=19)


def test_real_video_decoder_uses_centers_and_short_video_padding(tmp_path):
    import cv2
    path = tmp_path / 'video1.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 3, (32, 24))
    assert writer.isOpened()
    for level in [0, 100]:
        writer.write(np.full((24, 32, 3), level, dtype=np.uint8))
    writer.release()
    rgb, indices, count = read_video_frames(path)
    assert count == 2
    assert indices == [0, 1, 1, 1]
    assert rgb.shape == (4, 24, 32, 3)
    assert rgb[0].mean() < 5
    assert rgb[1].mean() > 90
    np.testing.assert_array_equal(rgb[1], rgb[3])


def test_wrapper_matches_official_pooling_and_cls_projection_on_small_model():
    # Execute the real official architectures at reduced dimensions. This verifies
    # the wrapper paths without requiring the multi-GB gated 1B checkpoint.
    model = RetrievalModel.__new__(RetrievalModel)
    torch.nn.Module.__init__(model)
    model.vision_encoder = PretrainInternVideo2(
        img_size=28, patch_size=14, embed_dim=32, depth=2, num_heads=4,
        attn_pool_num_heads=4, mlp_ratio=2, num_frames=4, clip_embed_dim=16,
        use_flash_attn=False, use_fused_rmsnorm=False, use_fused_mlp=False,
        clip_teacher_embed_dim=16, clip_teacher_final_dim=16, clip_return_layer=1,
        sep_image_video_pos_embed=True)
    cfg = BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=2,
                     num_attention_heads=4, intermediate_size=32, fusion_layer=1,
                     encoder_width=32)
    model.text_encoder = BertForMaskedLM(cfg)
    model.vision_proj = torch.nn.Linear(16, 8)
    model.text_proj = torch.nn.Linear(16, 8)
    model.eval()
    frames = torch.randn(1, 4, 3, 28, 28)
    tokens = {'input_ids': torch.tensor([[1, 2, 3, 0]]),
              'attention_mask': torch.tensor([[1, 1, 1, 0]])}
    with torch.inference_mode():
        _, pooled, _, _ = model.vision_encoder(frames.permute(0, 2, 1, 3, 4), None, False)
        expected_video = torch.nn.functional.normalize(model.vision_proj(pooled), dim=-1)
        output = model.text_encoder.bert(**tokens, return_dict=True, mode='text')
        expected_text = torch.nn.functional.normalize(model.text_proj(output.last_hidden_state[:, 0]), dim=-1)
    torch.testing.assert_close(model.video_features(frames), expected_video)
    torch.testing.assert_close(model.text_features(tokens), expected_text)
