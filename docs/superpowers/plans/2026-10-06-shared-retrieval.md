# Shared Retrieval Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; one fresh final reviewer.

**Goal:** Refactor the approved three-model baseline architecture and publish the tested code to the existing repository.
**Architecture:** Standard src/visual_retrieval package, model adapters over licensed implementations, shared feature bundle and dataset protocols. Preserve old entrypoints with thin wrappers; use explicit external roots.
**Tech Stack:** Python3.11, PyTorch2.5.1+cu121, Transformers4.40.2, NumPy, OpenCV, PyYAML.
**Spec:** docs/superpowers/specs/2026-10-06-shared-retrieval-design.md

## Global Constraints
- Data defaults S:\video_retrieval\msr_vtt, vatex, activitynet_captions. Repository data is reference-only.
- Corrected InternVideo2 CLS-only tokenizer and legacy cache compatibility remain intact.
- CLIP4Clip requires full trained meanP/2d checkpoint; no silent fallback.
- Comparison uses identical dataset items and queries, full-rank metrics, stable ties.
- Preserve prior command entrypoints and explicit CUDA requirements.

## Review Focus
- Package imports and entrypoints outside cwd.
- Checkpoint prefix/shape/head mismatch.
- Cross-model cache mixing and differing gallery/query text.
- Runtime path defaults versus reference annotations.
- Temporal sampling and feedback dimensions/zero vectors.

### Task1: Package migration and paths
- [ ] Write tests for external dataset roots and lightweight imports; run RED.
- [ ] Move functional modules/vendor code into src package, split dataset manifest and video sampler; add compatibility wrappers and packaging. Run old suite plus new path/import tests GREEN.

### Task2: Shared model and feature pipeline
- [ ] Write model pooling/checkpoint guard tests RED.
- [ ] Implement common encoder interface and CLIP/CLIP4Clip/InternVideo2 adapters. Generalize extraction/cache/evaluation with compatible InternVideo2 settings, configuration files, model-specific output defaults. Verify cached CLI pipelines and reduced architectures GREEN.

### Task3: Comparison and interaction
- [ ] Write fair-comparison and session/feedback tests RED.
- [ ] Add manifest preparation, baseline runner, exact search, comparison table and session evaluation commands. Implement tested query history/feedback logic; run full suite GREEN.

### Task4: Delivery
- [ ] Update docs, examples, downloader defaults and ignore rules. Validate real reference annotations using explicit paths only.
- [ ] Final fresh code review; fix significant findings with regression tests; full suite and CLI checks. Push with lease to main (user's continuing GitHub task authorization).
