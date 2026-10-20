# WebVid-CoVR: encode and evaluate

Run in PowerShell from `P:\visual-retrieval` after [installation](../README.md#install-once)
and [downloading samples](webvid_covr_download.md).
**Encodes CLIP, CLIP4Clip, InternVideo2 and CoVR-2; evaluates only CoVR-2.**

## 1. Prepare weights once

Get [CLIP4Clip weights](baselines.md#1-get-clip4clip-weights-once) and
[CoVR-2 weights](covr2.md#1-download-weights-once). Expected locations:

| File | Folder |
|---|---|
| `clip4clip_webvid150k_hf.bin` | `S:\video_retrieval\models` |
| `blip2_finetune_coco.pth`, `coir+covr.ckpt` | `S:\video_retrieval\weights\covr2` |

CLIP and InternVideo2 download automatically when absent.

## 2. Check and encode

Stop the demo first to free VRAM.

```powershell
.\.venv\Scripts\python.exe scripts\run_webvid_covr.py `
  --root "S:\video_retrieval\webvid_covr" --stage check

.\.venv\Scripts\python.exe scripts\run_webvid_covr.py `
  --root "S:\video_retrieval\webvid_covr" --stage extract `
  --device cuda --precision fp16 --decode-workers 4
```

Uses all downloaded test samples and the weight locations above. Models load one
at a time. Rerunning resumes valid caches. A different checkpoint location needs
`--clip4clip-checkpoint`, `--covr2-base-checkpoint` or `--covr2-checkpoint`.

## 3. Evaluate saved CoVR-2 features

```powershell
.\.venv\Scripts\python.exe scripts\run_webvid_covr.py `
  --root "S:\video_retrieval\webvid_covr" --stage evaluate `
  --validation-workers 4 --gallery-cache-mb 512
```

Uses CPU, loads no model, and shows gallery/query progress. Validation runs in
parallel; the RAM cache avoids repeated candidate reads when the gallery fits.
Scoring still computes every query/candidate pair and can take time. Ctrl+C keeps
encoded features; restart this command to evaluate again. Keep source media for
cache validation. Omitting `--stage` runs check → extraction → evaluation together.

## Options

| Need | Option |
|---|---|
| First 5 downloaded samples | `--limit 5` |
| Another split/output | `--split val`, `--output-root PATH` |
| Fewer baseline encoders | `--models clip` (CoVR-2 still runs) |
| Include reference videos as candidates | `--gallery all` (default: unique targets) |
| Save more ranked candidates | `--top-k 20` (full ranks/metrics unchanged) |
| Larger extraction batches | `--video-batch-size 8`, `--covr2-frame-batch-size 2` |
| Reduce evaluation RAM | `--gallery-cache-mb 128` (0 disables caching) |
| Slow disk contention | `--validation-workers 2` (0 is serial) |
| Deliberately allow omissions | `--allow-incomplete` (review failures first) |

Keep split/limit/subset/output/gallery options consistent between stages. Batch
settings preserve caches; changed model/precision/preprocessing needs a new output.
InternVideo2 still infers one video at a time. Lower batches after GPU OOM.

## Results

Default base: `S:\video_retrieval\webvid_covr\features\webvid_covr\test`.
Model folders hold separate features; `results\covr2\` contains
`evaluation.json`, `query_ranks.csv`, and `predictions.jsonl`.
Metrics: R@1/5/10/50 (%), MRR, mean/median rank. Reference videos are excluded per
query; duplicate targets share a candidate; ties retain gallery order. Partial
runs report exclusions and requested/evaluated counts.

This evaluates the [runtime CoVR-2 adaptation](covr2.md#method), not the official
benchmark. Five samples are a smoke test; high-K recall on tiny galleries is trivial.
These extracted features are not automatically registered in the demo.
