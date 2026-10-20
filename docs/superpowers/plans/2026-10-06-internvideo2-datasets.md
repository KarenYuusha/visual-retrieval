# Archive: InternVideo2 dataset integration (2026-10-06)

Historical implementation summary. Current commands: [baselines](../../baselines.md).

- Added combined MSR-VTT/VATEX galleries and ActivityNet caption segments.
- Shared adapters preserve interval sampling, corrected CLS-only text encoding and caches.
- Extraction/evaluation reject missing labels and incompatible provenance; partial work is resumable.
- Kept external dataset roots, CUDA pins and old entrypoints.
- Validation covered adapters, timestamps, label alignment and cached pipelines.
