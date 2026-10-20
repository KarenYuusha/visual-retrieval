# Archive: CoVR-2 verification notes (2026-10-16)

Current usage: [CoVR-2 guide](../../covr2.md).

- Upstream CoVR revision: `dd8a1a54e965d743314ce305eb1b8d64ed4e5860`.
- Released query/gallery weights passed CPU smoke checks; tiny Q-Former matched composition.
- Runtime scoring uses frozen-text max-token ITC, softmax / 0.1, weighted frame tokens,
  per-token normalization and mean-token dot product. It does not reuse annotation scores.
- Fixed corrupt-cache omission, EOF handling and Gradio null gallery-selection updates.
- Initial full CPU/browser suite: 138 passed. Canvas tests used VP9 because the test
  browser lacked H.264; FFmpeg separately verified production H.264 media.
- Submit-only follow-up removed draft-change search callbacks; 34 focused tests and
  three browser checks passed. Enter/Search commits image/text routing.
- CUDA VRAM/latency and full-dataset relevance remain hardware-side checks.
