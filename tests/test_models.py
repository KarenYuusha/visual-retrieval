import numpy as np
import pytest
import torch
from visual_retrieval.models.base import pool_frame_features
from visual_retrieval.models.clip4clip_encoder import extract_clip_weights, build_trained_clip


def test_frame_normalization_precedes_mean_pooling():
    frames = torch.tensor([[100., 0.], [0., 1.]])
    result = pool_frame_features(frames)
    torch.testing.assert_close(result, torch.tensor([[2**-.5, 2**-.5]]))


def test_clip4clip_requires_trained_clip_prefix_and_rejects_temporal_heads():
    with pytest.raises(ValueError, match='clip-prefixed'):
        extract_clip_weights({'visual.conv1.weight': torch.ones(1)})
    with pytest.raises(ValueError, match='meanP'):
        extract_clip_weights({'clip.visual.conv1.weight': torch.ones(1), 'frame_position_embeddings.weight': torch.ones(1)})


def test_clip4clip_full_checkpoint_is_loaded_without_fallback():
    from visual_retrieval.models.clip4clip_vendor.module_clip import CLIP
    model = CLIP(16, 28, 1, 64, 14, 8, 20, 64, 1, 1)
    original = {f'clip.{k}': v for k, v in model.state_dict().items()}
    loaded = build_trained_clip({'state_dict': {f'module.{k}': v for k, v in original.items()}}, require_vit_b32=False)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name].float(), value.float(), rtol=1e-3, atol=1e-3)
    incomplete = dict(original)
    del incomplete['clip.text_projection']
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        build_trained_clip(incomplete, require_vit_b32=False)


def test_clip_adapter_executes_small_real_transformers_model(tmp_path):
    import json
    from transformers import CLIPConfig, CLIPModel, CLIPTokenizer, CLIPImageProcessor, CLIPProcessor
    from visual_retrieval.models.clip_encoder import ClipEncoder
    cfg=CLIPConfig(projection_dim=512,
        text_config=dict(vocab_size=49408,hidden_size=64,intermediate_size=128,num_hidden_layers=1,
                         num_attention_heads=1,max_position_embeddings=77,bos_token_id=49406,eos_token_id=49407,pad_token_id=49407),
        vision_config=dict(hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=1,
                           image_size=224,patch_size=32))
    CLIPModel(cfg).save_pretrained(tmp_path)
    (tmp_path/'vocab.json').write_text(json.dumps({**{f'unused{i}</w>':i for i in range(1,49406)},'a</w>':0,'<|startoftext|>':49406,'<|endoftext|>':49407}))
    (tmp_path/'merges.txt').write_text('#version: 0.2\n')
    tokenizer=CLIPTokenizer(tmp_path/'vocab.json',tmp_path/'merges.txt')
    CLIPProcessor(CLIPImageProcessor(),tokenizer).save_pretrained(tmp_path)
    with pytest.raises(ValueError,match='ViT-B/32'):
        ClipEncoder(tmp_path,'cpu')
    adapter=ClipEncoder(tmp_path,'cpu',require_vit_b32=False)
    video=adapter.encode_video(np.zeros((12,32,48,3),dtype=np.uint8))
    text=adapter.encode_text(['a','a'])
    assert video.shape==(1,512) and text.shape==(2,512)
    np.testing.assert_allclose(np.linalg.norm(video,axis=1),1,atol=1e-5)
    np.testing.assert_allclose(text[0],text[1],atol=1e-5)
    from safetensors.torch import load_file, save_file
    weights=load_file(tmp_path/'model.safetensors')
    del weights['visual_projection.weight']
    save_file(weights,tmp_path/'model.safetensors',metadata={'format':'pt'})
    with pytest.raises(ValueError,match='missing'):
        ClipEncoder(tmp_path,'cpu',require_vit_b32=False)


def test_clip4clip_cannot_infer_smaller_model_from_missing_layers():
    shapes={'visual.conv1.weight':(768,3,32,32),'visual.proj':(768,512),
            'text_projection':(512,512),'token_embedding.weight':(49408,512),
            'positional_embedding':(77,512)}
    checkpoint={f'clip.{key}':torch.empty(shape,device='meta') for key,shape in shapes.items()}
    with pytest.raises(ValueError,match='12'):
        build_trained_clip(checkpoint)
