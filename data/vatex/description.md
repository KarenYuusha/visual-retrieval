# VATEX reference subset

- Recorded download: 1,000 train + 1,000 validation clips; seed 67.
- Runtime evaluation combines clips and retains all English captions.
- Source: YouTube, up to 360p, H.264, no audio; yt-dlp + Deno/Firefox cookies.
- Recorded settings: 3 workers, 2 concurrent encodes, 300s timeout, 2 GB source cache.
- Runtime root: `S:\video_retrieval\vatex`.

This folder is reference-only. Runtime `subset.json` lists validated downloads.
[Dataset format](../../docs/datasets.md) · [Run baselines](../../docs/baselines.md).
