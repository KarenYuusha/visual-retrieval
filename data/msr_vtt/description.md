# MSR-VTT reference subset

- Recorded selection: 1,000 train + 1,000 test videos; seed 67.
- Runtime evaluation combines selected videos and retains every caption.
- Annotations: `MSRVTT_data.json`, `sentences[].video_id` / `caption`.
- Runtime root: `S:\video_retrieval\msr_vtt`; videos: `raw_videos\video<ID>.mp4`.

This folder is reference-only. Runtime `subset.json` defines the actual selection.
[Dataset format](../../docs/datasets.md) · [Run baselines](../../docs/baselines.md).
