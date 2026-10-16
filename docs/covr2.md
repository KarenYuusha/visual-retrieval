# Image + text video search with CoVR-2

Attach one still image, then describe the change you want: for example, a frame
of a person walking + “the person is running on a beach”. The demo selects
CoVR-2 automatically and ranks up to 100 videos, 20 per page by default. Removing
the image restores the last text retrieval model. Text-only search retains its
variable result count. CoVR-2 requires its own gallery extraction; existing CLIP,
CLIP4Clip and InternVideo2 embeddings are not interchangeable with these tokens.

## Install and download on the GPU PC

Stop the running demo, then use the existing environment:

```powershell
cd P:\visual-retrieval
git pull
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -r requirements-demo.txt
.\.venv\Scripts\python.exe scripts\download_covr2.py --output "S:\video_retrieval\weights\covr2"
```

If using pip, replace the `uv pip install` line with
`.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt`.
The download is approximately 10.7 GB: frozen BLIP-2 COCO weights and the authors'
`coir+covr.ckpt`. Both files are verified with SHA-256; the Hub revision is pinned.
Do not install the upstream training environment or salesforce-lavis over the
project environment. Minimal inference components are included with their
licenses. The BERT tokenizer downloads on the first model load.

## Extract a pilot gallery

Use an existing extraction manifest to retain exactly its video/segment IDs.
This MSR-VTT example uses the CLIP manifest:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_covr2.py `
  --root "S:\video_retrieval\msr_vtt" `
  --source-manifest "S:\video_retrieval\msr_vtt\features\clip_vit_b32_12f_mean\manifest.json" `
  --base-checkpoint "S:\video_retrieval\weights\covr2\blip2_finetune_coco.pth" `
  --limit 100
```

If your CLIP features use a custom directory, point `--source-manifest` at that
actual `manifest.json`; another model's valid manifest also works. ActivityNet
requires a segment manifest with source paths and start/end times.

Run the demo locally against that pilot:

```powershell
.\.venv\Scripts\python.exe scripts\demo.py --model clip `
  --covr2-base-checkpoint "S:\video_retrieval\weights\covr2\blip2_finetune_coco.pth" `
  --covr2-checkpoint "S:\video_retrieval\weights\covr2\coir+covr.ckpt"
```

Open the local URL, search by text, choose a video, pause at a frame, and try
**Copy frame → click the search box → Ctrl+V → type a modification → Search**.
You can also upload or drop a still PNG, JPEG, WebP or BMP image. Only one image
is accepted; reference videos and animated images are rejected.

**Use frame** directly attaches the displayed frame and preserves typed text.
Use it when the browser denies clipboard access. Clipboard copy requires a
browser supporting PNG ClipboardItem writes and a secure context (localhost or
the HTTPS share link). The capture uses the current displayed video frame, not
the result thumbnail. For ActivityNet, the confirmation time includes the source
segment's start offset. Known copied/attached source videos are excluded from
that session's composed results; arbitrary uploaded images have no assumed
source video identity.

## Extract all three galleries

After checking the pilot, stop the demo to free GPU memory and run:

```powershell
$base = "S:\video_retrieval"
$weights = "$base\weights\covr2"
foreach ($folder in @("msr_vtt", "vatex", "activitynet_captions")) {
  .\.venv\Scripts\python.exe scripts\prepare_covr2.py `
    --root "$base\$folder" `
    --source-manifest "$base\$folder\features\clip_vit_b32_12f_mean\manifest.json" `
    --base-checkpoint "$weights\blip2_finetune_coco.pth"
  if ($LASTEXITCODE -ne 0) { throw "Extraction failed for $folder" }
}
```

Caches are stored under each dataset's `features\covr2`, with 15 frame centers
and 32 × 256 tokens per frame. Storage is roughly 246 KB per video/segment, plus
filesystem overhead. Extraction uses batch one; progress reports each saved or
skipped item. Rerunning resumes completed items. Changed source metadata,
intervals, or source file size/mtime invalidate the affected item. Different
checkpoint or precision settings require a separate/cleared CoVR-2 output cache.
Missing/failed videos are omitted and counted; they do not become empty vectors.

Use default CUDA extraction for CUDA serving. Explicit `--device cpu` extracts
fp32 and requires CPU serving; precision mismatches are rejected.

## Share with the laptop

Keep the existing username/password environment variables, or set them in the
same PowerShell window. Then launch with both checkpoint options:

```powershell
$env:VR_DEMO_USERNAME = "your-username"
$env:VR_DEMO_PASSWORD = "replace-with-your-password"
.\.venv\Scripts\python.exe scripts\demo.py --share `
  --covr2-base-checkpoint "S:\video_retrieval\weights\covr2\blip2_finetune_coco.pth" `
  --covr2-checkpoint "S:\video_retrieval\weights\covr2\coir+covr.ckpt"
```

Only one retrieval engine is resident at a time. Switching between text and
composed search unloads the previous engine, so the first query after a switch
will take longer. Query inference uses the trained CoVR-2 vision/Q-Former and a
frozen COCO text branch; gallery extraction uses the original COCO BLIP-2 model.
Full checkpoint loading and inference were smoke-tested on CPU. CUDA memory,
latency and end-to-end dataset relevance still need checking on your PC; the
12 GB GPU target uses fp16 and batch one.

## Method and limits

Implementation follows [CoVR-2 (paper v4)](https://arxiv.org/abs/2308.14746v4),
the [authors' code](https://github.com/lucas-ventura/CoVR/tree/dd8a1a54e965d743314ce305eb1b8d64ed4e5860)
and [released checkpoint](https://huggingface.co/lucas-ventura/CoVR2).
It preserves image-conditioned Q-Former composition, per-token normalization,
text-dependent frame weighting at temperature 0.1, and token-average dot-product
ranking. Final token averages are not renormalized as cosine vectors.

The authors' evaluation uses annotation-provided frame scores. This interactive
adaptation computes runtime frame relevance using standard BLIP-2 max-token
image/text similarity to the modification text. It does not claim reproduction
of the paper's benchmark numbers. Scores are ranking scores, not probabilities;
CoVR-2 returns the best 100 available candidates even for an ambiguous query.
With a smaller pilot or source exclusion it returns fewer. Gallery scoring
streams cached tokens from disk, so large galleries and slow disks increase
search time.
