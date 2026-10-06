"""CLIP ViT-B/32 frame-pooling baseline."""
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor
from .base import VideoTextEncoder, resolve_precision, autocast_context, pool_video_batch

MODEL_ID = 'openai/clip-vit-base-patch32'
PREPROCESSING = 'clip_b32_rgb_resize_crop224_12centers_frame_l2_mean_text77_v1'


class ClipEncoder(VideoTextEncoder):
    def __init__(self, checkpoint, device='cuda', precision='auto', tokenizer=None, require_vit_b32=True):
        self.device = device
        self.effective_precision = resolve_precision(device, precision)
        self.model, loading = CLIPModel.from_pretrained(str(checkpoint), output_loading_info=True)
        if loading['missing_keys'] or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise ValueError(f'CLIP checkpoint has missing or incompatible weights: {loading}')
        self.model = self.model.eval().requires_grad_(False).to(device)
        self.processor = CLIPProcessor.from_pretrained(str(tokenizer or checkpoint))
        config = self.model.config
        if config.projection_dim != 512 or config.vision_config.patch_size != 32:
            raise ValueError('The CLIP baseline requires ViT-B/32 with 512-dimensional projections.')
        if require_vit_b32:
            vision, text = config.vision_config, config.text_config
            if (vision.hidden_size != 768 or vision.num_hidden_layers != 12 or vision.num_attention_heads != 12
                    or vision.image_size != 224 or text.hidden_size != 512 or text.num_hidden_layers != 12
                    or text.num_attention_heads != 8 or text.vocab_size != 49408 or text.max_position_embeddings != 77):
                raise ValueError('Expected full ViT-B/32 vision and text architecture; local directory differs.')

    @torch.inference_mode()
    def encode_video(self, rgb_frames):
        return self.encode_videos([rgb_frames])

    @torch.inference_mode()
    def encode_videos(self, clips):
        inputs = self.processor(images=[Image.fromarray(frame) for clip in clips for frame in clip], return_tensors='pt')
        with autocast_context(self.device, self.effective_precision):
            features = self.model.get_image_features(pixel_values=inputs['pixel_values'].to(self.device))
        return pool_video_batch(features, [len(clip) for clip in clips]).cpu().numpy()

    @torch.inference_mode()
    def encode_text(self, texts):
        inputs = self.processor(text=texts, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
        with autocast_context(self.device, self.effective_precision):
            features = self.model.get_text_features(input_ids=inputs['input_ids'].to(self.device),
                                                    attention_mask=inputs['attention_mask'].to(self.device))
        return F.normalize(features.float(), dim=-1).cpu().numpy()
