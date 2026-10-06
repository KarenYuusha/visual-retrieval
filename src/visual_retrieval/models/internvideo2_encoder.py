"""Shared adapter over the verified InternVideo2 implementation."""
from .base import VideoTextEncoder
from .internvideo2_core import load_model, encode_text_batch, prepare_frames


class InternVideo2Encoder(VideoTextEncoder):
    def __init__(self, checkpoint, device='cuda', precision='auto', tokenizer='google-bert/bert-large-uncased'):
        self.device = device
        self.model, self.tokenizer, self.effective_precision = load_model(checkpoint, device, precision, tokenizer)

    def encode_video(self, rgb_frames):
        return self.model.video_features(prepare_frames(rgb_frames).to(self.device)).cpu().numpy()

    def encode_text(self, texts):
        return encode_text_batch(self.model, self.tokenizer, texts, self.device)
