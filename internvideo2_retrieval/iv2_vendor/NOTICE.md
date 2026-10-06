# Third-party source

Source: https://github.com/OpenGVLab/InternVideo
Commit: 3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3
License: Apache-2.0 (included as LICENSE).

These files come from InternVideo2/multi_modality:
- internvideo2.py: models/backbones/internvideo2/internvideo2.py
- pos_embed.py: models/backbones/internvideo2/pos_embed.py
- xbert.py: models/backbones/bert/xbert.py
- config_bert_large.json: configs/config_bert_large.json

Changes to internvideo2.py: removed imports of optional FlashAttention/fused
extensions and replaced native attention matrix materialization with PyTorch
scaled_dot_product_attention. Parameter names and model architecture are preserved.
The wrapper only computes the backbone and pooling path used for retrieval,
omitting execution of the training-only distillation decoders.

pos_embed.py, xbert.py, and config_bert_large.json are unchanged.
