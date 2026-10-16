import numpy as np
import pytest
import torch


def test_incomplete_checkpoint_is_rejected_before_random_weights_can_be_used(tmp_path):
    from visual_retrieval.models.covr2_encoder import read_state, load_required
    p = tmp_path / 'incomplete.pt'
    torch.save({'model': {'layer.weight': torch.ones(2,2)}}, p)
    state = read_state(p)
    layer = torch.nn.Linear(2,2)
    with pytest.raises(ValueError, match='missing'):
        load_required(layer, state, 'layer.')


def test_composed_query_matches_upstream_qformer_forward_and_is_image_conditioned():
    from visual_retrieval.models.covr2_encoder import composed_tokens
    from visual_retrieval.models.covr2_vendor.Qformer import BertConfig, BertLMHeadModel
    torch.manual_seed(7)
    cfg = BertConfig(vocab_size=101, hidden_size=24, num_hidden_layers=2,
                     num_attention_heads=4, intermediate_size=48, hidden_dropout_prob=0,
                     attention_probs_dropout_prob=0)
    cfg.encoder_width, cfg.add_cross_attention, cfg.cross_attention_freq, cfg.query_length = 16, True, 1, 2
    qformer = BertLMHeadModel(cfg).eval()
    proj = torch.nn.Linear(24, 8).eval()
    image = torch.randn(1, 4, 16)
    learnable = torch.randn(1, 2, 24)
    ids = torch.tensor([[3, 4, 5]])
    mask = torch.ones_like(ids)
    actual = composed_tokens(qformer, proj, learnable, image, ids, mask)
    expected = qformer.bert(ids, query_embeds=learnable,
        attention_mask=torch.ones(1, 5, dtype=torch.long), encoder_hidden_states=image,
        encoder_attention_mask=torch.ones(1,4,dtype=torch.long), return_dict=True).last_hidden_state[:, :2]
    expected = torch.nn.functional.normalize(proj(expected), dim=-1)
    torch.testing.assert_close(actual, expected)
    changed = composed_tokens(qformer, proj, learnable, image * -3, ids, mask)
    assert not torch.allclose(actual, changed, atol=1e-7)
