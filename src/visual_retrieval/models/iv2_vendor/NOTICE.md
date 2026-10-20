# InternVideo2 source notice

Source: [OpenGVLab/InternVideo](https://github.com/OpenGVLab/InternVideo), revision
`3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3`. License: [Apache-2.0](LICENSE).
Paths below are relative to upstream `InternVideo2/multi_modality/`:

| Local file | Upstream path |
|---|---|
| `internvideo2.py` | `models/backbones/internvideo2/internvideo2.py` |
| `pos_embed.py` | `models/backbones/internvideo2/pos_embed.py` |
| `xbert.py` | `models/backbones/bert/xbert.py` |
| `config_bert_large.json` | `configs/config_bert_large.json` |

Only `internvideo2.py` changed: optional FlashAttention/fused imports removed;
attention uses PyTorch scaled_dot_product_attention. Architecture/parameter names
are preserved. The wrapper runs retrieval backbone/pooling, skipping training-only
distillation decoders. Other listed files are unchanged.
