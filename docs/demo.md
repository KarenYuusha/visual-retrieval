# Run the demo

Run in PowerShell from `P:\visual-retrieval` after [installation](../README.md#install-once).
Requires [saved baseline features](baselines.md) and source videos.

## Local PC

```powershell
.\.venv\Scripts\python.exe scripts\demo.py --check
.\.venv\Scripts\python.exe scripts\demo.py
```

Open <http://127.0.0.1:7860>. Choose a model/dataset, type a query, then press
**Enter** or **Search**. Typing does not refresh results. Click a thumbnail to play.
The grid adjusts its columns automatically and shows 20 results per page.

Text results use a score threshold, so counts vary and may be zero.

**WebVid-CoVR:** after [encoding](webvid_covr_evaluation.md), restart the demo and
select **WebVid-CoVR** in the dataset list. Default discovery uses
`S:\video_retrieval\webvid_covr\features\webvid_covr\test` for all four model spaces.
For another split/output, edit `configs/demo.yaml` baseline paths and
`covr2_features` (relative to the dataset root), then launch with `--config`.

## Image + text with CoVR-2

Prepare [both weights and the gallery](covr2.md), then launch:

```powershell
.\.venv\Scripts\python.exe scripts\demo.py --allow-unknown-precision `
  --covr2-base-checkpoint "S:\video_retrieval\weights\covr2\blip2_finetune_coco.pth" `
  --covr2-checkpoint "S:\video_retrieval\weights\covr2\coir+covr.ckpt"
```

Select **WebVid-CoVR** (or another prepared dataset), paste/upload a reference
image, describe the change, then press **Enter/Search**. CoVR-2 selects automatically.
You can also pause a result and click **Use frame**. Text-only submissions return
to text retrieval. The first composed query loads the model and may take longer.
The precision flag is only needed for older `auto` caches missing dtype metadata.

## Laptop on another network

Set credentials and launch in the same PC terminal:

```powershell
$env:VR_DEMO_USERNAME = "your-username"
$env:VR_DEMO_PASSWORD = "replace-with-your-password"
.\.venv\Scripts\python.exe scripts\demo.py --share
```

Open the printed `https://…gradio.live` link on the laptop and log in.
Keep the PC awake and its terminal running. Ctrl+C stops the demo.

## Useful options

| Need | Add to `scripts\demo.py` |
|---|---|
| Port 7860 is occupied | `--port 7861` |
| Start with CLIP | `--model clip` |
| Different data base | `--base-root "D:\video_retrieval"` |
| Custom dataset registry | `--config configs\demo.yaml` (edit its paths first) |
| Open UI before model loading | `--no-warmup` |
| Change page size | `--page-size 30` |
| Adjust text match count | `--min-score 0.20 --score-margin 0.05` |
| CPU inference | `--device cpu` |

Use either `--base-root` or `--config`. All-dataset search requires matching
checkpoint, tokenizer, preprocessing and extraction precision across galleries.
If incompatible, select one dataset or re-extract matching features.
`--check` prints the datasets and fields that differ. Requested `auto` and explicit
precision can combine when their recorded actual precision matches; unknown
historical precision is not treated as a verified match.

For older `auto` caches missing actual precision, append
`--allow-unknown-precision` to the launch command. This permits combined search
only when all other model settings match and recorded precisions agree. The demo
uses a gallery with recorded precision for query encoding and labels the legacy
precision unverified. Re-extract matching caches for strictly verified comparisons.

Model switches load one GPU encoder at a time and may take a while.
ActivityNet previews play only the retrieved segment. Media caches are under
`S:\video_retrieval\demo_cache`; stop the app before deleting them to reclaim space.
