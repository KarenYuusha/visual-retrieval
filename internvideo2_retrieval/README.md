# InternVideo2 compatibility entrypoints

The implementation now lives in `src/visual_retrieval`, with shared CLIP/CLIP4Clip/InternVideo2 extraction and evaluation. See the [root README](../README.md) for installation, all-model comparison and dataset paths.

These scripts retain previous commands (`extract_features.py`, `evaluate_retrieval.py`, `search_video.py`, `run_datasets.py`). The corrected CLS-only tokenizer, licensed vendor code and matching InternVideo2 cache settings are preserved.

Default dataset locations now follow the user's layout: `S:\video_retrieval\msrvtt`, `S:\video_retrieval\vatex`, `S:\video_retrieval\activitynet`. The repository data folder is reference-only. Existing data under older names is still usable with explicit root overrides.

Install requirements from the repository root. The old local requirements also include the dependencies needed by these compatibility commands.
