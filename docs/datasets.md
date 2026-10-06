# Runtime datasets

The defaults are `S:\video_retrieval\msr_vtt`, `S:\video_retrieval\vatex`, and `S:\video_retrieval\activitynet_captions`. Repository `data/` annotations are reference-only. No runtime command falls back to them.

Each root needs `subset.json` containing video-ID string lists under `train`, `test`, `val`, or `validation`. IDs are combined with first-occurrence deduplication; seed/size/target fields are ignored. MSR-VTT retains its original train+test selection. Every selected item must have a nonempty caption.

Videos are `raw_videos/<ID>.mp4`. ActivityNet can also resolve filenames without the `v_` prefix. The canonical manifest records item ID, source video ID, path and optional interval; queries record caption text and the correct item ID.

| Dataset | Annotation files under raw_data/ or root | Schema |
|---|---|---|
| MSR-VTT | MSRVTT_data.json | sentences rows: video_id, caption, optional sen_id |
| VATEX | vatex_training_v1.0.json + vatex_validation_v1.0.json | list of videoID and enCap arrays |
| ActivityNet | train.json + val_1.json | object keyed by video ID, with sentences/timestamps |

`--annotations PATH [PATH ...]` overrides extraction annotation discovery. ActivityNet accepts `--activitynet-val-file val_2.json` instead of val_1; never mix conflicting versions implicitly.

ActivityNet segment IDs are `v_ID__seg0000`. Decoder samples only actual presentation times within `[start,end)`, including variable-frame-rate MP4s. Short intervals repeat the final frame. An interval after the last decoded frame fails; an oversized end uses the available frames. Missing/unusable timestamp timelines fail explicitly, with a transcode suggestion.

Canonical manifests can be exported for inspection with `scripts/prepare_manifests.py`. Extraction consumes the same dataset adapter directly and records hashes of the subset/annotation files, so a manifest export is not a prerequisite. Fair comparisons validate actual saved query/item identities.
