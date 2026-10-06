# Project structure

The class-project baseline milestone is implemented as an installable `visual_retrieval` package using the conventional `src/visual_retrieval` layout.

| Module | Responsibility |
|---|---|
| data/adapters.py + manifests.py | Canonical gallery/query records for all datasets |
| data/video_sampling.py | Shared interval-aware RGB sampling |
| models/base.py + registry.py | Encoder contract, precision and checkpoint provenance |
| models/*_encoder.py | CLIP, trained CLIP4Clip meanP, InternVideo2 |
| features/cache.py + extraction.py | Atomic resumable feature generation |
| retrieval/index.py + search.py | Exact cosine search and matching query encoder |
| evaluation/ranking_metrics.py | Full-rank R@1/R@5/R@10/MRR |
| evaluation/baseline_evaluation.py + comparison.py | Saved-feature scoring and matched cross-model comparisons |
| interaction/ | Query history, query refinement, relevance feedback |
| evaluation/session_evaluation.py | Per-turn ranks, metrics and latency for labeled sessions |
| cli/ + scripts/ | Dataset preparation and model/dataset orchestration |
| legacy/ | Preserved CLIP HDF5 entrypoints |

Model adapters expose `encode_video(rgb_frames)` and `encode_text(texts)`. Video inputs are RGB uint8[T,H,W,3]; outputs are normalized float32[B,512]. Model-specific preprocessing remains inside the adapter. NumPy scoring and lightweight dataset preparation do not import model implementations.

Milestones:
1. Three-model baseline extraction and comparison on three datasets (current).
2. Search UI with playback and model selection (next).
3. Hook up accumulated queries and positive/negative feedback (backend already available).
4. Curated, held-out multi-turn evaluation sessions and final experiments/report.

No empty UI placeholders are shipped. Future `app/` can call the tested GalleryIndex/RetrievalSession APIs. Changing the active model requires a fresh session/compatible query encoder; never mix embeddings between models.
