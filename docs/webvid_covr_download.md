# Download WebVid-CoVR samples

Run in PowerShell from `P:\visual-retrieval` after [installation](../README.md#install-once).
Needs FFmpeg/ffprobe and network access; no GPU is needed.

## Download 5 samples

```powershell
.\.venv\Scripts\python.exe download_data\download_webvid_covr.py `
  --root "S:\video_retrieval\webvid_covr" --split test --samples 5 --source hub
Start-Process "S:\video_retrieval\webvid_covr\samples\test\preview.html"
```

One sample is a reference-video / modification-text / target-video triplet.
Five samples need up to ten videos. The preview shows reference frames and targets.

## Download more

Repeat with `--samples 1000`. Valid downloaded videos are reused.

| Option | Meaning |
|---|---|
| `--samples N` | Requested complete triplets, not video count |
| `--split test` | Default; also `val`/`train` (availability varies) |
| `--source hub` | Authors' mirror; useful for test. `auto` also tries original URLs |
| `--start-row 5` | Start at another zero-based annotation row |
| `--max-attempts 100` | Bound attempted rows; default `max(50, samples*10)` |
| `--timeout 30 --retries 2` | Network controls |

Output under the chosen root: `raw_videos\`, `reference_frames\`, `raw_data\`,
and `samples\test\{subset.json,annotations.csv,preview.html}`.
`subset.json` records selected IDs, paths, source revisions and failures.
Ctrl+C preserves completed downloads. A shortfall exits with code 1 and saves a
partial preview; rerun or inspect the failures. Train annotations alone are about
437 MB, so start with test.

Next: [encode videos and evaluate only CoVR-2](webvid_covr_evaluation.md).
Sources: [annotations](https://huggingface.co/datasets/lucas-ventura/WebVid-CoVR) ·
[video mirror](https://huggingface.co/datasets/lucas-ventura/WebVid).
