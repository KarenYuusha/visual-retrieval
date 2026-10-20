# Archive: shared retrieval design (2026-10-06)

Historical requirements. Current usage: [baselines](../../baselines.md).

- Compare CLIP/CLIP4Clip (12 frames) and InternVideo2 (4 frames) using shared adapters.
- External dataset roots; ActivityNet segments; corrected InternVideo2 tokenizer.
- Separate provenance-checked model caches; strict trained CLIP4Clip meanP/2d loading.
- Match gallery/query identities and full-rank protocols before comparison.
- Preserve compatibility entrypoints and legacy HDF5 tools without silently converting caches.
- Optional history/feedback backend; UI was outside this original milestone.
