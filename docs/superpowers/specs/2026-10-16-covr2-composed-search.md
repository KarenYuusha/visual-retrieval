# Archive: CoVR-2 composed search design (2026-10-16)

Historical requirements. Current usage: [CoVR-2](../../covr2.md).

- One still reference image + modification text returns up to 100 videos, paginated.
- Shared composer supports upload/paste and copy/attach of the displayed video frame.
- Frozen COCO gallery encoder, trained composed Q-Former, separate 15-frame token caches.
- Preserve per-token normalization, frame weighting and token-average scoring.
- Respect segment bounds, exclude known reference sources and reject stale UI events.
- Keep text retrieval available, model spaces separate and one GPU engine active.
