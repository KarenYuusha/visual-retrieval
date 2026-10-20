# Models and weights

| Model | Checkpoint | Frames / text limit |
|---|---|---|
| CLIP | `openai/clip-vit-base-patch32` | 12 / 77 tokens |
| CLIP4Clip | `Searchium-ai/clip4clip-webvid150k` | 12 / 32 tokens |
| InternVideo2 | `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` | 4 / 40 tokens |
| CoVR-2 | Frozen BLIP-2 COCO + trained `coir+covr.ckpt` | 15 gallery frames / 64 query-text tokens |

[Baseline download/run commands](baselines.md) · [CoVR-2 setup](covr2.md).
CLIP and InternVideo2 auto-download on first extraction; CoVR-2 needs both weights.

CLIP/CLIP4Clip normalize frame features, mean-pool and normalize the video.
InternVideo2 jointly encodes four frames with trained projections and its corrected
CLS-only tokenizer. All baseline vectors are 512-D and ranked by cosine similarity.
CoVR-2 uses separate frame/query tokens and query-dependent scoring.

CLIP4Clip accepts trained ViT-B/32 **meanP, 2d** checkpoints in original or complete
HF CLIP format. Temporal heads, 3d patches, missing tensors and base-CLIP fallback
are rejected. Searchium weights were trained on WebVid-150k; identify that source
in reports. No upstream clone or FlashAttention compilation is needed.

Keep checkpoint/tokenizer/preprocessing/precision provenance matched. These are
combined-subset experiments, not official benchmark reproductions; record any
training/evaluation dataset overlap.

Sources: [CLIP](https://github.com/openai/CLIP) ·
[CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) ·
[InternVideo2](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2/multi_modality) ·
[CoVR-2](https://github.com/lucas-ventura/CoVR).
Vendored revisions, changes and licenses are recorded in each model's `NOTICE.md`.
