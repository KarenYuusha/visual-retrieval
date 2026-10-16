# SDD ledger — plan: docs/superpowers/plans/2026-10-16-covr2-composed-search.md

Execution: existing isolated linked worktree, inline implementation; user approved
the design and said proceed. Upstream commit dd8a1a54e965d743314ce305eb1b8d64ed4e5860.

Pre-flight: model tensor shapes feed scorer and cache; service result metadata
must preserve the existing preview/page contract. GPU lifetime is shared by the
two search paths, not independently retained by two services.

Model/scoring: strict required-key and shape loading, tensor-only mmap checkpoint
loading with an inert scheduler allowlist; actual released weights loaded and
both query [32,256] and gallery [15,32,256] CPU smoke checks passed. Tiny actual
Q-Former forward agrees with upstream composition and responds to image changes.
Gallery uses original COCO BLIP-2, query uses trained CoVR-2 vision/Q-Former.

Scoring decision: upstream BLIP-2 gallery annotation scores are precomputed; its
public get_scores helper handles BLIP-1 2D features only. Runtime scoring here
uses standard BLIP-2 max-token ITC similarity to frozen text CLS, softmax / 0.1,
frame-weighted token pooling, per-token normalization, then mean-token dot
product. This is the paper-consistent runtime adaptation, not a reproduction of
annotation-provided scores or a claim to reproduce paper benchmark metrics.

Gallery extraction: 15 actual-frame centers, segment-aware shared decoding,
atomic per-item caches with source/interval invalidation, checkpoint SHA-256
identity and precision validation. Interrupted runs reuse valid completed items.
Text/CoVR-2 services unload the prior engine before loading the next one.

UI: shared image/text composer, image-based model routing, current paused-frame
Canvas capture, PNG clipboard and direct attachment fallback; copied image pixel
hashes retain known source identity for exclusion. Existing text-search browser
checks passed (2 cases). Chromium test build lacks H.264; canvas browser test uses
VP9 previews while separate media tests verify the production H.264 delivery.

Independent reviewer found a corrupt-token cache edge case. Fixed per-item
startup omission, runtime omission after file corruption, and EOF handling in
resumable extraction. Regression tests cover one bad item beside a valid item,
live corruption, and truncated-cache regeneration. Reviewer found no remaining
critical/important issue in model/scorer/cache/UI readthrough.

Browser testing exposed a Gradio 6.9 null-update issue: gr.update removes the
explicit selected_index=None property, so clicking the same card index on a new
page does not emit select. Return a Gallery component update instead to preserve
null and reset the selected index. The composed browser check deliberately clicks
the first card on both pages and also checks image removal/model restoration.

Final verification (CPU environment):
`GRADIO_ANALYTICS_ENABLED=False VR_BROWSER_TEST=1 OMP_NUM_THREADS=1 /tmp/vr-covr-env/bin/python -m pytest -q`
passed 138 tests, including all three browser integrations, with no skipped tests
(50.84 seconds). Existing dependency deprecation warnings remain. Released
checkpoint query and 15-frame gallery smoke tests passed separately. CLI help
for download/prepare/demo and git diff --check passed. Focused final review of
the Gallery null-selection fix found no blockers. CUDA VRAM/latency and relevance
on the user's full datasets remain PC-side validation, not claimed here.

Implementation file names were consolidated into composed/prepare.py,
demo/composed.py and demo/frames.py; the interfaces and behavior match the spec.
Windows download, pilot/full extraction, launch and sharing steps are recorded
in docs/covr2.md. Publication is against remote main parent
5a7bb13fb7e8c769546a35051fa37774163d5113, whose tree matches the local base.
