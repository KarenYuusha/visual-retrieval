# CoVR-2: image + text search

Run in PowerShell from `P:\visual-retrieval` after [installation](../README.md#install-once).
Stop the demo before extraction to free VRAM.

## 1. Download weights once

```powershell
.\.venv\Scripts\python.exe scripts\download_covr2.py --output "S:\video_retrieval\weights\covr2"
```

Downloads about 10.7 GB: `blip2_finetune_coco.pth` and `coir+covr.ckpt`.
Use the existing project environment; no LAVIS installation is needed.

## 2. Prepare galleries

Requires existing [baseline manifests](baselines.md). This prepares all three datasets:

```powershell
$base = "S:\video_retrieval"
foreach ($folder in @("msr_vtt", "vatex", "activitynet_captions")) {
  .\.venv\Scripts\python.exe scripts\prepare_covr2.py `
    --root "$base\$folder" `
    --source-manifest "$base\$folder\features\clip_vit_b32_12f_mean\manifest.json" `
    --base-checkpoint "$base\weights\covr2\blip2_finetune_coco.pth"
  if ($LASTEXITCODE -ne 0) { throw "Extraction failed for $folder" }
}
```

For a pilot, append `--limit 100` to the Python command. Change the manifest
path if your baseline output is elsewhere. ActivityNet needs segment metadata.
Output: each dataset's `features\covr2` (~246 KB per item in fp16).
Reruns resume valid tokens; changed sources/intervals invalidate affected items.
CUDA extraction/serving uses fp16; CPU uses fp32 and needs matching caches.

Optional speed controls: `--decode-workers 4 --frame-batch-size 2`.
Default frame batch is 1; reduce it if GPU memory runs out.

## 3. Launch and search

```powershell
.\.venv\Scripts\python.exe scripts\demo.py --model clip `
  --covr2-base-checkpoint "S:\video_retrieval\weights\covr2\blip2_finetune_coco.pth" `
  --covr2-checkpoint "S:\video_retrieval\weights\covr2\coir+covr.ckpt"
```

1. Search by text and pause a result at the desired frame.
2. Click **Use frame**, or **Copy frame** then paste into the search box.
3. Type a modification and press **Search**. CoVR-2 selects automatically.

You can also upload one still image. **Use frame** works if clipboard access is
blocked. Removing the image restores text search on the next submission.
Known reference sources are excluded; up to 100 candidates are paginated.
For laptop access, set [demo credentials](demo.md#laptop-on-another-network) and
append `--share` to the launch command above.

## Method

[Paper](https://arxiv.org/abs/2308.14746v4) · [Code](https://github.com/lucas-ventura/CoVR/tree/dd8a1a54e965d743314ce305eb1b8d64ed4e5860) · [Weights](https://huggingface.co/lucas-ventura/CoVR2).
Frozen COCO BLIP-2 encodes 15 gallery frames; trained CoVR-2 composes image/text.
Runtime modification-text frame weights use temperature 0.1, followed by normalized
token pooling and mean-token dot products. This differs from official annotation
frame scores; subset results are not a reproduction of the paper's benchmark.
CoVR-2 tokens are separate from CLIP/CLIP4Clip/InternVideo2 embeddings.
