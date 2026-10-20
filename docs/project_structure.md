# Code map

| Path under `src/visual_retrieval/` | Purpose |
|---|---|
| `data/` | Dataset adapters, canonical manifests, segment-aware decoding |
| `models/` | Encoders, precision, checkpoint validation, licensed vendor code |
| `features/` | Baseline extraction, resumable caches, decoder prefetch |
| `composed/` | CoVR-2 gallery/query extraction and frame-token scoring |
| `retrieval/` | Baseline query encoding and cosine search |
| `demo/` | Gradio UI, dataset registry, engine lifecycle, frame/media actions |
| `evaluation/` | Baseline, CoVR-2, comparison and multi-turn metrics/exports |
| `interaction/` | Optional query history/refinement and feedback backend |
| `cli/` | Workflow orchestration |
| `legacy/` | Original CLIP HDF5 tools |

`scripts/` contains command entrypoints; `configs/` holds model/dataset profiles.
`internvideo2_retrieval/` preserves older command names. Legacy HDF5 caches are
not automatically converted to the shared NPZ format.

Baseline encoders return normalized `[B,512]` vectors. CoVR-2 stores separate
`[15,32,256]` gallery tokens and image/text queries. Never mix model spaces.
Only one retrieval engine occupies the GPU at a time.
See the [README task index](../README.md#start-here) for commands.
