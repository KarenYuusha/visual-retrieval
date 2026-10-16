# CoVR-2 Composed Search Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, with one final review.

**Goal:** Add image/text composed video search and video-frame reuse to the existing Gradio demo.

**Architecture:** A separate CoVR-2 encoder and per-frame feature bundle feed a
query-dependent scorer. A shared demo dispatcher selects text or composed search
from the presence of an image. The multimodal composer and frame actions reuse
the existing previews, pagination and session guards.

**Tech Stack:** Python, PyTorch, released CoVR-2/BLIP-2 weights, Gradio 6.9, browser Canvas/Clipboard APIs.

**Spec:** `docs/superpowers/specs/2026-10-16-covr2-composed-search.md`

## Global Constraints
- Image reference only; one image plus nonempty modification text.
- Fixed composed top 100; page size 20; no visible Top K control.
- 15 target frames; ActivityNet bounds respected; separate resumable caches.
- Preserve existing models, CUDA dependency pins, and authenticated remote access.
- Never compare CoVR-2 query vectors with existing model galleries.

## Review Focus
- Pasted image while typing retains the text and invalidates stale search results.
- Copied frame is the displayed frame, including relative segment time.
- Pending search/page/playback events cannot replace a newer query or reference.
- Target encoder and composed Q-Former weights are loaded strictly and used on their correct branches.
- Short, missing, corrupt or incomplete media never yields misaligned feature IDs.

## Tasks
1. Model and scorer: add `models/covr2_encoder.py`, the minimal licensed upstream
   inference modules, and `composed/scoring.py`. Interfaces: encode_query(image,
   text), encode_frames(frames), encode_scoring_text(text); features retain
   frame/query-token axes. Test exact upstream normalization/weighting, strict
   checkpoint loading and query-vs-target branch separation before implementation.
2. Gallery extraction/service: add `composed/prepare.py`, `demo/composed.py`
   and `scripts/prepare_covr2.py`. Reuse dataset adapters and interval
   sampling. Store IDs, source signatures, sample indices/counts and encoder hashes.
   Test resume, invalidation, partial galleries, source exclusion, ties and top K.
3. Composer/frame integration: add `demo/composed.py` and `demo/frames.py` (embedded browser capture functions),
   modify `demo/app.py` and configuration. Test image validation, routing, engine
   unloading, text preservation, source metadata, stale events and clipboard fallback.
4. End-to-end verification and guide: numerical smoke test with released weights
   when obtainable, full suite, browser copy/paste/page/playback tests, independent
   review, Windows extraction/launch instructions, and publication.

Each task follows RED -> GREEN -> targeted verification -> commit. The final
suite includes all existing tests; record hardware/network limits explicitly.
