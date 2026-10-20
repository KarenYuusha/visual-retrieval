# Archive: CoVR-2 implementation (2026-10-16)

Historical plan summary. [Design](../specs/2026-10-16-covr2-composed-search.md) ·
[Verification notes](2026-10-16-covr2-progress.md) · [Current guide](../../covr2.md).

- Added strict BLIP-2/CoVR-2 inference and a separate query-dependent token scorer.
- Added 15-frame, segment-aware, provenance-checked gallery caches.
- Added image/text routing, frame copy/attachment, pagination and stale-event guards.
- Kept model spaces separate, one GPU engine and existing CUDA pins.
- Tested upstream numerical behavior, cache corruption, browser interaction and resume.
