# Interactive video retrieval

Class-project code for comparing **CLIP**, **CLIP4Clip**, and **InternVideo2**, then evaluating query refinement and relevance feedback. No training is performed by these tools.

Runtime datasets live outside the repository:

| Dataset | Default root | Gallery/query protocol |
|---|---|---|
| MSR-VTT | `S:\video_retrieval\msrvtt` | Combined train+test videos, every caption |
| VATEX | `S:\video_retrieval\vatex` | Combined train+validation clips, every English caption |
| ActivityNet Captions | `S:\video_retrieval\activitynet` | Combined train+validation timestamped segments, one sentence per segment |

**The repository's `data/` directory is reference-only. No command selects it automatically.** Each runtime root must contain `subset.json`, `raw_videos/`, and annotation files under `raw_data/` (or the root). See [dataset formats](docs/datasets.md).

## Install on Windows (Python 3.11, NVIDIA GPU)

Run from the cloned repository root:

```powershell
git pull
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected GPU build: `2.5.1+cu121`, `12.1`, `True`. The package requires explicit CUDA builds of Torch and torchvision. CPU execution requires `--device cpu`; there is no silent fallback. These requirements target Windows/Linux, not macOS.

For pip: `py -3.11 -m venv .venv`, then `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

## Check data and run baselines

```powershell
# File/annotation checks only; no checkpoint download or video decoding
.\.venv\Scripts\python.exe scripts\run_baselines.py --stage check

# Optional: export canonical manifests for inspecting items/queries
.\.venv\Scripts\python.exe scripts\prepare_manifests.py

# Run the two automatically available models on all three datasets
.\.venv\Scripts\python.exe scripts\run_baselines.py --models clip internvideo2

# Run all three: CLIP4Clip requires your trained meanP/2d ViT-B/32 checkpoint
.\.venv\Scripts\python.exe scripts\run_baselines.py --clip4clip-checkpoint "S:\video_retrieval\models\clip4clip_meanP.bin"
```

The final checkpoint path is an example: supply the real file. CLIP4Clip accepts official `clip.*` weights in a plain state dictionary or `model`/`module`/`state_dict` wrapper. Temporal (`seqTransf`/`seqLSTM`), cross-encoder and 3d-patch checkpoints are rejected by this meanP adapter. It never substitutes base CLIP weights. See [model details and checkpoint requirements](docs/models.md).

CLIP and InternVideo2 download weights/tokenizers on first extraction. If InternVideo2 access is gated, accept the conditions on its model page and authenticate with Hugging Face. Local overrides: `--clip-checkpoint DIR`, `--internvideo2-checkpoint FILE`.

### Run one model/dataset

```powershell
.\.venv\Scripts\python.exe scripts\extract_features.py --config configs\clip.yaml --dataset vatex
.\.venv\Scripts\python.exe scripts\extract_features.py --model clip4clip --dataset msrvtt --checkpoint "S:\video_retrieval\models\clip4clip_meanP.bin"
.\.venv\Scripts\python.exe scripts\extract_features.py --model internvideo2 --dataset activitynet
```

Model profile YAML values are defaults; explicit CLI values override them. `--data-root PATH` changes a single dataset location. `--annotations PATH [PATH ...]` supplies explicit annotation files. `--text-batch-size 8` can reduce memory use.

The all-model runner accepts `--base-root`, `--msrvtt-root`, `--vatex-root`, `--activitynet-root`, `--datasets`, and `--models`. `configs/datasets.yaml` documents the canonical external layout. Outputs can be moved with `--output-root PATH`; each dataset/model gets its own subfolder.

ActivityNet defaults to `--activitynet-mode segments`. Use `--activitynet-mode video` for one whole-video embedding and one concatenated paragraph query; long paragraphs are truncated by each model's tokenizer. Use a separate output directory/root when changing protocols. `activitynet_captions` remains an accepted alias in the extraction CLI.

## Saved features and evaluation

Default feature folders are inside each external dataset:

| Model | Folder under `<dataset>\features\` |
|---|---|
| CLIP | `clip_vit_b32_12f_mean` |
| CLIP4Clip | `clip4clip_vit_b32_12f_meanP` |
| InternVideo2 | `internvideo2_stage2_1b_all` |

Each folder contains normalized float32 `features.npz` (`video_features`, `video_ids`, `text_features`, `text_video_ids`), `queries.json`, `manifest.json`, `run_config.json`, resumable `video_cache/`, `text_cache.npz`, and `failed_videos.json`.

```powershell
# Evaluate saved features: no GPU or model needed
.\.venv\Scripts\python.exe scripts\evaluate_baselines.py --features-dir "S:\video_retrieval\vatex\features\clip_vit_b32_12f_mean"

# Reevaluate all three saved model outputs; no checkpoints needed
.\.venv\Scripts\python.exe scripts\run_baselines.py --stage evaluate

# Enforce matched galleries and query text/labels before comparing
.\.venv\Scripts\python.exe scripts\run_baselines.py --stage compare
```

Reports include extraction wall time, peak allocated GPU memory (when CUDA is used),
feature bundle size and cache reuse counts. Cached extraction timings are labeled by
the reused item count; compare fresh extraction runs when reporting throughput.
Reports contain R@1/R@5/R@10 as percentages and MRR on a 0–1 scale using full-gallery ranks. Ties use gallery order. Evaluation saves `evaluation.json` and per-query `query_ranks.csv` in each feature directory. The runner saves aggregate results to `S:\video_retrieval\reports\baseline_comparison.json`; `--summary PATH` overrides it.

Comparisons require identical gallery order, query IDs, caption text, targets and protocol, and complete galleries. They fail explicitly rather than comparing different subsets. For a standalone JSON+CSV table use `scripts\compare_models.py --features-dirs DIR1 DIR2 DIR3 --output PATH.json`.

Extraction reuses valid caches; source-video changes invalidate individual embeddings. Model/data/preprocessing changes require a new output directory. Missing files stop extraction by default. `--allow-missing` permits a partial extraction, but it exits with an error after saving its partial bundle. Retry to complete it. Standalone evaluation can deliberately score it using `--allow-incomplete`; cross-model comparison requires completeness.

## Search and interaction

```powershell
.\.venv\Scripts\python.exe scripts\search.py --features-dir "S:\video_retrieval\msrvtt\features\internvideo2_stage2_1b_all" --query "a man carrying an umbrella"
```

Search loads the matching checkpoint from saved provenance. Results include source-video paths and segment timestamps. `--data-root` relocates playback paths.

The interaction package includes latest/accumulated query history and normalized relevance feedback, with multi-turn session evaluation. See [session format and experiments](docs/interaction.md). A graphical UI is a later milestone; this refactor delivers the shared baseline and interaction backend.

## Code structure and compatibility

Implementation lives in `src/visual_retrieval/`: `data/`, `models/`, `features/`, `retrieval/`, `interaction/`, `evaluation/`, and `cli/`. `scripts/` contains thin entrypoints. `configs/` holds baseline profiles. See [architecture and milestones](docs/project_structure.md).

Old `internvideo2_retrieval/` commands are compatibility wrappers over the shared package; its corrected tokenizer and cache configuration are retained. Runtime folder defaults now use the names above. Existing files under an older folder name remain usable with an explicit `--data-root` or runner root override.

The original CLIP HDF5 scripts still work through `extract_clip_features.py` and `evaluate_clip_retrieval.py`. Their implementation lives in `src/visual_retrieval/legacy/`. Old `.h5` files are not silently converted/reused by the new NPZ pipeline. Use the new shared pipeline for matched comparisons.

## Validation

```powershell
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover legacy InternVideo2 behavior, cache-only CLIP/CLIP4Clip/InternVideo2 extraction, real reduced model architectures, timestamped decoding, stable ranks, matching comparison inputs, external defaults, entrypoints outside the repository, query refinement, and feedback timing.

Your full checkpoints and S: drive videos cannot be executed in this development workspace. No real dataset scores are fabricated. These are combined-subset experiments rather than official benchmark scores. Document whether each downloaded checkpoint was trained on any of your evaluation datasets; no local fine-tuning does not establish zero-shot evaluation.
