# Archive: WebVid-CoVR evaluation design (2026-10-19)

Historical requirements. Current usage: [WebVid-CoVR evaluation](../../webvid_covr_evaluation.md).

- Encode CLIP, CLIP4Clip, InternVideo2 and CoVR-2; evaluate only CoVR-2.
- Consume downloader subset IDs, reference JPEGs, modification texts and target labels.
- Separate model caches, provenance checks, resumable extraction and one GPU engine.
- Rank unique targets by default; optionally include references; exclude each query's source.
- Export full ranks, R@1/5/10/50, MRR and mean/median ranks with stable ties.
- Strict missing-target handling; partial runs record exclusions and denominators.
- Runtime frame weighting differs from official annotation scores; no benchmark-parity claim.
