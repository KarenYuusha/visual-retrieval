# Encode WebVid-CoVR and evaluate only CoVR-2

`scripts/run_webvid_covr.py` consumes the examples produced by the
[WebVid-CoVR downloader](webvid_covr_download.md). It runs sequentially:

| Step | Encoded inputs | Evaluation |
|---|---|---|
| CLIP | All unique reference/target videos + modification texts | None |
| CLIP4Clip | Same videos and texts | None |
| InternVideo2 | Same videos and texts | None |
| CoVR-2 | Frozen BLIP-2 gallery tokens + trained reference-image/text query tokens | Full-rank retrieval metrics |

No baseline evaluator or baseline comparison is called. Baseline embeddings are
saved for future use; they are not combined with the CoVR-2 features. CoVR-2 uses
the downloader's middle-frame reference JPEG and the paired modification text.
No training or fine-tuning is performed.

## Run on the GPU PC

Stop the demo first to free GPU memory. Use the existing project environment;
there are no new package dependencies or changes to the CUDA pins.

```powershell
cd P:\visual-retrieval
git pull

# Validate downloaded sample IDs and file paths; no model loading or downloads:
.\.venv\Scripts\python.exe scripts\run_webvid_covr.py `
  --root "S:\video_retrieval\webvid_covr" --stage check

# Encode all four model spaces, then evaluate ONLY CoVR-2:
.\.venv\Scripts\python.exe scripts\run_webvid_covr.py `
  --root "S:\video_retrieval\webvid_covr" `
  --device cuda --precision fp16 `
  --clip4clip-checkpoint "S:\video_retrieval\models\clip4clip_webvid150k_hf.bin" `
  --covr2-base-checkpoint "S:\video_retrieval\weights\covr2\blip2_finetune_coco.pth" `
  --covr2-checkpoint "S:\video_retrieval\weights\covr2\coir+covr.ckpt"
```

CLIP and InternVideo2 use the existing Hugging Face cache or download their model
files when absent. Optional `--clip-checkpoint PATH` (local HF directory) and
`--internvideo2-checkpoint PATH` (local `.pt`) select explicit weights. CLIP4Clip
requires the trained meanP/2d checkpoint, not plain CLIP weights. The CoVR-2
weights are the same two files used by the composed demo. Defaults for those and
CLIP4Clip match the explicit paths above under the dataset root's parent.

The default split is `test`. It uses all samples currently saved in
`ROOT/samples/test/subset.json`: if you downloaded five, it processes those five.
Use `--limit 5` to restrict a larger saved subset to its first five examples.
`--split val` or `train` selects the corresponding downloaded subset.

Models are loaded one at a time and released before loading the next model.
CLIP/CLIP4Clip use the established 12-frame pipeline; InternVideo2 uses four
frames. CoVR-2 gallery extraction uses 15 frame centers. Its image/text query
encoder is loaded after the gallery encoder is released. CoVR-2 uses fp16 on
CUDA and fp32 on CPU; `--precision` configures the baseline encoders only.

## Tune extraction throughput

CPU decoding now overlaps model inference using a bounded thread pool. Models
still run sequentially, with all GPU inference on the main thread. Defaults are
two decoding workers, four videos per CLIP/CLIP4Clip inference batch, one video
for InternVideo2, and one frame per CoVR-2 gallery inference batch.

For the 12 GB GPU, start with this command (checkpoint paths use the defaults above):

```powershell
.\.venv\Scripts\python.exe scripts\run_webvid_covr.py `
  --root "S:\video_retrieval\webvid_covr" `
  --device cuda --precision fp16 `
  --decode-workers 4 --video-batch-size 4 --covr2-frame-batch-size 2
```

Increase `--video-batch-size` to 8 if CLIP/CLIP4Clip have spare VRAM. InternVideo2
retains its conservative per-video inference fallback; this setting buffers its
inputs but does not batch its GPU forward pass. Increase
`--covr2-frame-batch-size` separately to 4 only if VRAM permits. If CUDA runs out
of memory, lower the relevant batch size and rerun to resume completed items.
There is no automatic retry with smaller batches.

Use `--decode-workers 2` if four workers contend for disk bandwidth or RAM;
`--decode-workers 0` restores synchronous decoding. Keep source videos and output
caches on an SSD where possible. More workers are not necessarily faster.
The decoder retains only a bounded number of upcoming sources, rather than
loading the entire dataset into RAM.

The standalone `scripts/prepare_covr2.py` also accepts `--decode-workers 4` and
`--frame-batch-size 2` for existing MSR-VTT, VATEX or ActivityNet manifests.
Worker and batch settings do not change cache identity, so tuning them reuses
valid existing video/token caches. Batched floating-point inference may differ
slightly numerically from single-item inference.

Extraction manifests record the selected settings and elapsed seconds; baseline
manifests also record inference batch counts and new/reused video counts. Compare
elapsed time on the same uncached inputs (use separate `--output-root` folders
for comparisons). A fully cached rerun measures cache validation, and a five-sample
run is dominated by model loading. Evaluation-only still uses CPU scoring of saved
CoVR-2 features; these extraction controls do not move evaluation onto the GPU.

## Extract and evaluate separately

Append `--stage extract` to the full command to save all features without metrics.
Then evaluate the saved CoVR-2 features:

```powershell
.\.venv\Scripts\python.exe scripts\run_webvid_covr.py `
  --root "S:\video_retrieval\webvid_covr" --stage evaluate
```

Evaluation-only uses NumPy on CPU. It loads no model and needs no baseline or
CoVR-2 checkpoint files, but retains and checks saved feature provenance. Keep the
selected raw target videos and reference JPEGs available for cache validation.

`--models clip clip4clip internvideo2` controls baseline **extraction** (all three
by default). CoVR-2 extraction is always included. This option never enables
baseline evaluation. Example: `--models clip` encodes CLIP and CoVR-2 only.

## Candidate gallery and metrics

Default `--gallery targets` ranks every query against the unique target video IDs
in the selected subset. Duplicate target labels share one candidate video.
`--gallery all` also includes all downloaded reference videos as candidates.
Each query's own reference video is excluded. An example whose reference ID equals
its target ID is rejected. The evaluator scores every eligible gallery candidate;
there is no text-baseline prefilter or top-100 restriction.

Reports contain Recall@1/5/10/50 (percent), MRR, median rank and mean rank. Target
ranks are one-based, with ties resolved by the recorded gallery order. Recall is
the fraction of evaluated queries whose correct target ranks within K; MRR is the
mean reciprocal target rank. Higher recall/MRR and lower mean/median rank are
better. `--top-k 10` controls only the candidates exported for inspection; it does
not limit the ranks used for evaluation.

**Five examples are a pipeline smoke test, not a reliable accuracy estimate.**
With five target candidates, R@5/R@10/R@50 are trivially 100% for valid queries.
The report records the exact gallery size, actual per-query candidate counts,
query counts, exclusions, checkpoint provenance and subset hash.

This evaluates the **existing runtime CoVR-2 implementation**, not the official
full benchmark. Frame weights are computed from each modification text using
frozen BLIP-2 max-token image/text similarity, softmax temperature 0.1, followed
by weighted frame-token pooling and mean-token dot-product ranking. It does not
reuse the annotations' precomputed frame scores: those refer to the authors'
frame extraction, which differs from our current center sampling. Reference
JPEG extraction also follows our downloader. Do not compare these subset numbers
directly with the paper's published benchmark results.

## Saved files, resume and partial data

Default output:

```text
S:\video_retrieval\webvid_covr\features\webvid_covr\test\
  clip\features.npz
  clip4clip\features.npz
  internvideo2\features.npz
  covr2\tokens\*.npy
  covr2_queries\query_cache\*.npz
  results\covr2\evaluation.json
  results\covr2\query_ranks.csv
  results\covr2\predictions.jsonl
```

Baseline NPZ files contain aligned `video_ids`, normalized 512-dimensional
`video_features`, modification `text_features`, `text_video_ids`, `query_ids`
and `reference_ids`. Each model folder records its settings and extraction
manifest. CoVR-2 has its own 15 × 32 × 256 gallery tokens and per-query 32 × 256
composed tokens plus frozen scoring-text features. **No baseline metrics are
written.** CSV/JSONL reports preserve reference IDs, target labels, modification
texts, predicted IDs, full target ranks, scores and labelled top candidates.

Rerun extraction to reuse valid per-video/query caches. Changed media signatures,
reference image bytes, or modification text trigger re-encoding of affected items.
Different checkpoint/precision/preprocessing settings require a new
`--output-root PATH` or clearing the relevant incompatible feature cache.
Increasing the downloaded sample count reuses previously encoded items.

Missing, corrupt or stale features cause evaluation to stop by default. After
reviewing those failures, `--allow-incomplete` permits a partial run, explicitly
recording omitted candidate IDs, excluded queries and requested/evaluated counts.
Download status/failures and requested/downloaded/selected/evaluated sample counts
are retained separately. Overall `complete` remains false for partial downloads;
`evaluation_complete` describes whether all selected valid inputs were evaluated.
A query with a missing correct target is excluded, never silently assigned an
incorrect rank. A run with zero valid queries produces no accuracy report.

These features are not automatically registered as a new demo dataset. This
command is an extraction and CoVR-2-only evaluation pipeline.
