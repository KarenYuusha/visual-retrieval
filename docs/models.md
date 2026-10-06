# Baseline models and checkpoints

| Model | Configuration | Text processing |
|---|---|---|
| CLIP | openai/clip-vit-base-patch32, 12 sampled frames, normalize each frame then mean and normalize | CLIP tokenizer, maximum77tokens |
| CLIP4Clip | trained official ViT-B/32 meanP, 2d patches, 12 sampled frames; same normalized meanP pooling | CLIP tokenizer, maximum32tokens, BOS/EOS, ID0 padding |
| InternVideo2 | OpenGVLab/InternVideo2-Stage2_1B-224p-f4, 4 jointly encoded frames, learned pooling/projections | official caption cleaning, BERT maximum40tokens, CLS without trailing SEP |

Comparison uses cosine similarity without video-text reranking. Sampling and preprocessing are declared in saved manifests. This is a comparison of configurations, not an isolated architecture experiment. Uniform segment-center frame sampling is a project choice; do not equate these subset runs with official evaluation protocols.

CLIP4Clip requires an explicitly supplied trained checkpoint. `clip.*` encoder weights are loaded strictly; absent weights are not filled using base CLIP. Ordinary CLIP checkpoints without the trained wrapper prefix are rejected. Only meanP/2d is supported; seqLSTM, seqTransf, tightTransf and 3d-patch weights are rejected. Prefixes cannot prove a checkpoint was actually trained: document where your file came from and which dataset it used.

Sources:
- https://github.com/openai/CLIP
- https://github.com/ArrowLuo/CLIP4Clip
- https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2/multi_modality

Selected upstream CLIP4Clip source at508ffa3de39ba0563a03199c440ab602a72e9b6f is vendored under MIT. Selected InternVideo2 source at3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3 is vendored under Apache2.0. Each vendor directory includes license/notice. No official repo clone or FlashAttention compilation is required to run these adapters.

InternVideo2 uses its trained vision/text projection weights and validates missing retrieval parameters. Changing the corrected tokenizer will invalidate results. The original tokenizer-fix migration remains available for matching legacy cache settings.
