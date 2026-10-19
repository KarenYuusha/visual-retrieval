# Download a small WebVid-CoVR subset

One sample is a complete **reference video + modification text + target video**
triplet. Five samples can therefore require up to ten distinct videos. The script
also extracts each reference video's middle frame as a JPEG, suitable for trying
your image + text search, and writes a local preview page.

From the existing Windows project environment:

```powershell
cd P:\visual-retrieval
git pull
.\.venv\Scripts\python.exe download_data\download_webvid_covr.py --samples 5 --source hub
```

Default output: `S:\video_retrieval\webvid_covr`. To choose another location:

```powershell
.\.venv\Scripts\python.exe download_data\download_webvid_covr.py `
  --root "S:\video_retrieval\webvid_covr" --split test --samples 5
```

Open the preview after download:

```powershell
Start-Process "S:\video_retrieval\webvid_covr\samples\test\preview.html"
```

Each card shows a reference image, the modification, both original captions and
the target video. Full reference and target MP4s are under `raw_videos`. JPEGs are
under `reference_frames`. `samples\test\subset.json` records local paths and
selected row indices; `annotations.csv` there preserves the original selected CSV
rows, including official frame scores. The full source CSV stays in `raw_data`.

Increase the count later; valid cached videos are reused:

```powershell
.\.venv\Scripts\python.exe download_data\download_webvid_covr.py --samples 100
```

Samples are considered in official CSV order, starting at zero-based
`--start-row 0`. Counts refer to **complete triplets**, not MP4 files. Unavailable
pairs are logged and skipped until the target is reached or `--max-attempts` is
exhausted (default `max(50, samples*10)`). A shortfall produces a partial preview,
failure details in `subset.json`, and exit code 1. Ctrl+C preserves completed
videos and sample metadata; rerun the command to continue. Temporary transfers
are not accepted as complete MP4s. FFprobe checks stream metadata, then FFmpeg
decodes the video stream to reject truncated files with otherwise valid headers. Reference frames are regenerated per run from
the verified source files.

Additional controls:

- `--split test` (default), `val`, or `train`.
- `--source auto` (default): try the authors' Hugging Face mirror, then original URLs.
- `--source hub`: only use the mirror; useful for the test split.
- `--source original`: only use the original URL lists.
- `--max-attempts 10`: bound how many annotation rows are attempted.
- `--start-row 5`: start at a different zero-based row.
- `--timeout 30 --retries 2`: network controls.
- `--annotation-file PATH`: use an already downloaded official CSV.

The default test annotations are only about 665 KB. Training annotations are
about 437 MB even for a small training subset, so use the **test split for the
initial five-sample trial**. The script downloads individual selected MP4s, not
the entire video repository. `requests` and FFmpeg/ffprobe on PATH are required;
the existing project/demo environment already uses these. It requires Python
3.11+ and needs no GPU, MPI, or new model weights.

Sources: [official CoVR code](https://github.com/lucas-ventura/CoVR),
[annotations](https://huggingface.co/datasets/lucas-ventura/WebVid-CoVR),
[authors' video mirror](https://huggingface.co/datasets/lucas-ventura/WebVid).
Annotation and mirror revisions are pinned and recorded in `subset.json`, along
with the downloaded annotation's SHA-256. Some original video URLs are no longer
available. The mirror is intended for the test videos; train/val availability
through original URLs is not guaranteed. Network/proxy failures are recorded per row; the attempt limit
prevents an unbounded run when downloads keep failing.

This script downloads an inspection subset. It does not extract CoVR-2 features,
register a new demo gallery, or provide full-benchmark retrieval metrics. Those
require preparing the chosen gallery and the appropriate evaluation protocol.
