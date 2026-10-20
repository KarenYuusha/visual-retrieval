# Dataset paths and formats

Runtime base: `S:\video_retrieval`. Repository `data/` contains reference records,
not the active video database. [Run baseline extraction](baselines.md).

| Dataset folder | Annotations under `raw_data\` or root | Evaluation unit |
|---|---|---|
| `msr_vtt` | `MSRVTT_data.json`: `sentences` with `video_id`, `caption` | Video; all captions |
| `vatex` | `vatex_training_v1.0.json`, `vatex_validation_v1.0.json`: `videoID`, `enCap` | Clip; all English captions |
| `activitynet_captions` | `train.json`, `val_1.json`: sentences + timestamps keyed by video | Captioned segment |

Each root needs `subset.json` (ID lists under train/test/val/validation) and
`raw_videos\<ID>.mp4`. Selected splits are combined and IDs deduplicated.
ActivityNet IDs use `v_ID__seg0000`; frames/playback stay inside `[start,end)`.
Invalid/empty intervals fail rather than generating embeddings.

| Task | Command/option |
|---|---|
| Verify all runtime datasets | `.\.venv\Scripts\python.exe scripts\run_baselines.py --stage check` |
| Export canonical item/query manifests (optional) | `.\.venv\Scripts\python.exe scripts\prepare_manifests.py` |
| Change base folder | Runner: `--base-root PATH` |
| Change a single root | Runner: `--msrvtt-root PATH`, `--vatex-root PATH`, `--activitynet-root PATH` |
| Supply explicit annotations | `scripts\extract_features.py --annotations PATH [PATH ...]` |
| Use alternate ActivityNet validation file | `--activitynet-val-file val_2.json` |

Existing downloaders are `download_data/download_msrvtt.py`, `download_vatex.py`,
and `download_activitynet.py`; edit their top-level roots/sample settings before
running. MSR-VTT also needs `datasets`; VATEX needs yt-dlp, Deno and configured
browser cookies. For adjustable image/text triplets, use the separate
[WebVid-CoVR downloader](webvid_covr_download.md).
