# Text retrieval baselines

Run in PowerShell from `P:\visual-retrieval` after [installation](../README.md#install-once).
Uses MSR-VTT, VATEX and ActivityNet by default; see [dataset layout](datasets.md).

## 1. Get CLIP4Clip weights once

```powershell
New-Item -ItemType Directory -Force "S:\video_retrieval\models" | Out-Null
Invoke-WebRequest -Uri "https://huggingface.co/Searchium-ai/clip4clip-webvid150k/resolve/main/pytorch_model.bin" -OutFile "S:\video_retrieval\models\clip4clip_webvid150k_hf.bin"
```

CLIP and InternVideo2 download their weights/tokenizers on first extraction.
CLIP4Clip needs this trained checkpoint; plain CLIP weights are not a substitute.

## 2. Check datasets

```powershell
.\.venv\Scripts\python.exe scripts\run_baselines.py --stage check
```

Checks files/annotations without loading models or decoding videos.

## 3. Extract all three models

```powershell
.\.venv\Scripts\python.exe scripts\run_baselines.py --stage extract `
  --clip4clip-checkpoint "S:\video_retrieval\models\clip4clip_webvid150k_hf.bin"
```

Stop the demo first to free VRAM. Rerun the same command to resume valid caches.
To restrict a run, append `--datasets msrvtt --models clip` (no CLIP4Clip
checkpoint is needed when that model is excluded).

## 4. Evaluate saved features

```powershell
.\.venv\Scripts\python.exe scripts\run_baselines.py --stage evaluate
```

Uses CPU; no checkpoints are needed. For matched cross-model comparison:

```powershell
.\.venv\Scripts\python.exe scripts\run_baselines.py --stage compare
```

Both commands require all selected feature bundles. Use `--models`/`--datasets`
to select available ones. Comparison requires identical, complete galleries and queries.

## Outputs

Each dataset stores features under its own `features\` directory:

| Model | Folder |
|---|---|
| CLIP | `clip_vit_b32_12f_mean` |
| CLIP4Clip | `clip4clip_vit_b32_12f_meanP` |
| InternVideo2 | `internvideo2_stage2_1b_all` |

Each folder contains `features.npz`, manifests, resumable caches, and—after
evaluation—`evaluation.json`, `query_ranks.csv`, `predictions.jsonl`.
Runner summary: `S:\video_retrieval\reports\baseline_comparison.json`.
Metrics: full-gallery R@1/5/10 (%) and MRR (0–1); [inspect errors](error_analysis.md).
These combined-subset runs are not official benchmark scores.

## Common adjustments

| Need | Add to the extraction command |
|---|---|
| Different data base | `--base-root "D:\video_retrieval"` |
| Separate output base | `--output-root "S:\video_retrieval\experiments\run1"` |
| More decoding overlap | `--decode-workers 4` (default 2; 0 disables prefetch) |
| Larger CLIP/CLIP4Clip batch | `--video-batch-size 8` (default 4; InternVideo2 defaults to 1) |
| Less caption memory | `--text-batch-size 8` |
| Explicit half precision | `--precision fp16` |

Keep the same root/output overrides for later evaluation. InternVideo2 infers
videos individually. Lower batch size after out-of-memory; valid caches survive.
Changing weights/precision/preprocessing requires a new output folder.

For a deliberately partial bundle, evaluate that folder directly:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_baselines.py `
  --features-dir "S:\video_retrieval\activitynet_captions\features\clip4clip_vit_b32_12f_meanP" `
  --allow-incomplete --top-k 10
```

Review `failed_videos.json` first. `--top-k` limits exported candidates, not metric
ranks. Partial extraction saves completed work before reporting an error.

For a command-line search of saved features:

```powershell
.\.venv\Scripts\python.exe scripts\search.py `
  --features-dir "S:\video_retrieval\msr_vtt\features\clip_vit_b32_12f_mean" `
  --query "a man carrying an umbrella"
```

Search loads the matching model from saved provenance. For a grid/player, use the
[demo](demo.md).
Run any script with `--help` for less common options.
