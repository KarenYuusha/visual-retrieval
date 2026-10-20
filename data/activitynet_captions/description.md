# ActivityNet Captions reference subset

- Recorded download: 500 train + 500 validation videos; seed 67.
- Annotations: official `train.json` + `val_1.json`; source: TornadoLabs MP4 mirror.
- IDs retain `v_`; annotations provide duration, timestamps and sentences.
- Runtime default: one gallery item/query per captioned segment, combining splits.
- Runtime root: `S:\video_retrieval\activitynet_captions`.

This folder is reference-only. Runtime `subset.json` defines the actual selection.
[Dataset format](../../docs/datasets.md) · [Run baselines](../../docs/baselines.md).
