# Minimal Gradio Demo Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, with one fresh code review at completion. Steps use checkbox syntax.

**Goal:** Deliver the approved minimal text-to-video demo on the GPU PC with remote laptop access.

**Architecture:** Configurable gallery registry, one active encoder, exact combined
cosine index, lazy FFmpeg media cache and Gradio Blocks UI. Reuse existing model
loading/provenance and feature normalization, leaving extraction/evaluation intact.

**Tech Stack:** Python 3.11/3.12, NumPy, PyTorch, Gradio 6.9.0, FFmpeg.

**Spec:** ../specs/2026-10-13-gradio-demo-design.md

## Global Constraints

- Default data base S:\video_retrieval; repo data/ remains reference-only.
- Keep CUDA torch/torchvision exact pins; hub 0.36.2 supports transformers 4.40.2 and Gradio 6.9.0.
- Default model InternVideo2, gallery all, top_k 10; one live GPU encoder.
- Only serve generated media cache; sharing is opt-in and requires credentials.
- Preserve ActivityNet segment boundaries and gallery-order ties.

## Review Focus

- Partial/missing datasets must not silently imply a complete gallery.
- Colliding item IDs across configurable datasets must remain distinct.
- Combined models cannot mix checkpoint/tokenizer/preprocessing identities.
- Concurrent searches/model switches must not exceed the one-encoder contract.
- Invalid paths/timestamps and FFmpeg failures must not expose or return arbitrary files.

### Task 1: Configurable search and encoder lifecycle

**Files:** demo/config.py, demo/service.py, tests/test_demo_service.py, configs/demo.yaml.
**Interfaces:** DemoConfig.defaults/from_yaml -> dataset specs; SearchService.search(text, model, dataset='all', top_k=10) -> ranked metadata; warmup(model), close(), dataset_choices(model).

- [x] Write tests for namespaced combined ranking/filtering, config paths/custom data, missing bundles, incompatible checkpoint/precision/tokenizer/dimensions and one-encoder reuse/release.
- [x] Run tests and confirm missing implementation fails.
- [x] Implement registry, gallery validation/metadata and a lock around encoder loading and text inference.
- [x] Run service tests and baseline suite; document results.

### Task 2: Lazy browser media

**Files:** demo/media.py, tests/test_demo_media.py.
**Interfaces:** MediaCache.thumbnail(result) -> cached JPG; video(result) -> cached MP4; placeholder() -> image. Inputs are validated result metadata from Task 1.

- [x] Write real FFmpeg tests for trimmed intervals, cached reuse/source-change invalidation, missing files and malformed intervals.
- [x] Run tests and confirm missing implementation fails.
- [x] Implement atomic, per-key locked FFmpeg generation with timeouts and no shell execution.
- [x] Run tests and inspect thumbnail/decoded preview timeline.

### Task 3: Minimal UI and remote launcher

**Files:** demo/app.py, scripts/demo.py, optional demo dependency extra,
requirements-demo.txt, docs/demo.md, README, tests/test_demo_app.py.
**Interfaces:** build_demo(service, media, default_model, top_k) -> Blocks;
main() configures launch, local/share/auth/cache; CLI --check validates configured galleries and binaries without loading models.

- [x] Write UI config/auth/entrypoint/callback tests.
- [x] Implement minimal Blocks, explicit search, clearing stale results on selector changes, one selected player and remote credentials.
- [x] Render and exercise the real page with browser automation and synthetic fixture galleries; inspect screenshots.
- [x] Run the full suite, review the full change, fix verified findings, and push the reviewed tree preserving unrelated data edits.
- [x] Provide tested Windows install/launch commands and state hardware/network verification limits.

Validation: full suite 115 passed, opt-in authenticated browser integration passed.
Windows Python 3.11 dependencies resolved with explicit CUDA builds. Review
findings fixed with regressions for obsolete callback outputs, segment bounds and
effective precision. H.264 media delivery/FFprobe verified; browser decoding,
actual GPU and remote network remain user hardware checks.
