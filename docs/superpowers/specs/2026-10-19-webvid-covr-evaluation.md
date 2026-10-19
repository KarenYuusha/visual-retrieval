# WebVid-CoVR encoding and CoVR-2-only evaluation

User scope: encode downloaded WebVid-CoVR with CLIP, CLIP4Clip, InternVideo2;
encode the image + modification queries and gallery with existing CoVR-2;
evaluate only CoVR-2. No baseline evaluation/comparison calls or metrics.

Consume downloader samples/<split>/subset.json. Validate stable query IDs,
reference/target/image paths and labels. Encode all unique reference/target videos
with each baseline and save aligned modification-text vectors without composition.
CoVR-2 reuses frozen COCO 15-frame gallery extraction, trained image-conditioned
queries, and the existing runtime frame-weighted scorer. Model spaces stay separate.
One GPU engine at a time; resume with model, source and query provenance checks.

Default evaluation gallery is unique downloaded target IDs; --gallery all includes
reference videos too. Exclude each query's reference video, reject self-target
queries, rank the full candidate pool with deterministic gallery-order ties.
Report R@1/5/10/50, MRR, mean/median rank, per-query ranks and labelled top candidates.
Strict by default; --allow-incomplete explicitly reports exclusions and denominators.

This is subset evaluation of the existing runtime implementation, not reproduction
of official benchmark numbers. Runtime weights use modification-text BLIP-2 ITC;
annotation-provided scores are retained in the downloader but not reused against
our different frame sampling. Small galleries make higher-K recall trivial.
