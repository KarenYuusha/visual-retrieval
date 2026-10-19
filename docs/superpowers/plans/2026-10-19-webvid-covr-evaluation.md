# WebVid-CoVR Evaluation Implementation Plan

> **For agentic workers:** Execute inline with superpowers:executing-plans and one final review.

**Goal:** Encode four model spaces; evaluate only CoVR-2 on downloaded image/edit/target examples.
**Architecture:** Validated subset adapter, baseline feature extractor, composed-query cache,
full-gallery CoVR-2 evaluator, and sequential CLI orchestrator.
**Tech Stack:** Existing Python/PyTorch encoders, NumPy/Pillow, shared media decoder.
**Spec:** docs/superpowers/specs/2026-10-19-webvid-covr-evaluation.md

## Global Constraints
- No CLIP/CLIP4Clip/InternVideo2 evaluation or feature mixing.
- One active inference engine; existing CUDA dependency pins unchanged.
- Default target-only subset gallery, explicit all-video option and reference exclusion.
- Strict alignment/provenance; never silently omit missing ground truth.

## Review Focus
- Duplicate targets and references do not duplicate gallery entries.
- Cache reuse cannot survive changed images/text/source/model settings unnoticed.
- Full target rank computed beyond any exported top-k cutoff.
- Partial data retains requested vs evaluated denominators and explicit omissions.
- Evaluation-only loads saved CoVR-2 features without requiring baseline checkpoints.

## Tasks
- [x] RED/GREEN: validated subset and deterministic gallery/query contracts.
- [x] RED/GREEN: per-model video/text caches and CoVR-2 query caches; GPU release.
- [x] RED/GREEN: strict CoVR-2-only full-rank evaluation, exports and stage routing.
- [x] Real-media fixture integration, regression suite, review and Windows guide.

Ruling: reuse existing runtime CoVR-2 scoring; do not apply upstream annotation
frame scores to differently sampled frames or claim official benchmark parity.


Execution ledger: existing isolated linked worktree; user requested implementation.
New code is scoped to WebVid-CoVR adapters/extractors/evaluator/CLI. No baseline
evaluation import/call or metrics are added. Baseline encoding preserves separate
512-D model spaces; existing CoVR-2 gallery/query/scoring paths are reused.

RED/GREEN evidence: missing adapter, extractor, query-cache, evaluator and CLI
contracts failed before implementation. Fourteen targeted tests now pass. Golden
ranks include duplicate labels, source exclusion and rank 120 with top-1 export.
Real video fixture integration uses deterministic encoder stubs and weak-reference
tracking to verify stage routing, only one live encoder and fully cached reruns.
Evaluation-only succeeds after checkpoint removal; actual five-sample preflight
succeeds from outside the repo. This environment is CPU-only; no new four-model
GPU accuracy run is claimed.

Review rulings: require reference_frames/{reference_id}.jpg to prevent swapped
reference/target image leakage. Preserve downloader complete/failures and separate
requested/downloaded/selected/evaluated counts; overall complete remains false
for partial downloads even when all selected examples are evaluated. Both review
findings reproduced as failing tests and passed after fixes. Focused rereview
found no blockers.

Final verification: `GRADIO_ANALYTICS_ENABLED=False OMP_NUM_THREADS=1
/tmp/webvid-eval-env/bin/python -m pytest -q` passed 162 tests, with the three
opt-in browser tests skipped (no UI changes), in 44.36 seconds. CLI check/help,
compilation and whitespace checks passed. Independent review found no remaining
blockers after the two data-contract fixes. Publication base is remote main
506b1613ae5d5aa02405355190dd3f542f9fa7df, matching the local base tree.
