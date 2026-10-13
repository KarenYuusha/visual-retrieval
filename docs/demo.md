# Minimal text-to-video demo

The Gradio app runs on your GPU PC. Your laptop opens a password-protected
temporary share link in its browser. No GPU, Python environment, model weights or
dataset copy is needed on the laptop. This version searches text queries; image
plus text composition is a later extension.

## Install on the PC

Use the existing repository environment (Python 3.11 or 3.12):

```powershell
cd P:\visual-retrieval
git pull
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -r requirements-demo.txt
```

For pip instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
```

The optional demo extra installs Gradio 6.9.0. The project updates Hugging Face
Hub to 0.36.2, compatible with Gradio and Transformers 4.40.2. CUDA PyTorch
2.5.1+cu121 and torchvision 0.20.1+cu121 remain explicitly pinned. Install through
this requirements file so the CUDA wheel index is available.

FFmpeg and ffprobe must be installed and available on PATH:

```powershell
ffmpeg -version
ffprobe -version
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Check saved galleries

```powershell
.\.venv\Scripts\python.exe scripts\demo.py --check
```

The check reads saved video embeddings and metadata, reports available galleries
and omitted item counts, and checks FFmpeg availability. It does not load encoders,
decode videos, or start a web server. Raw videos are checked when previews are
requested. Missing/invalid feature directories are reported and excluded; at least
one valid gallery is required.

Default roots:

| Dataset | Root |
|---|---|
| MSR-VTT | `S:\video_retrieval\msr_vtt` |
| VATEX | `S:\video_retrieval\vatex` |
| ActivityNet Captions | `S:\video_retrieval\activitynet_captions` |

The app uses each root's model-specific `features/` directories, including partial
bundles saved by extraction. Available models appear in the dropdown. Failed or
omitted gallery items are not searchable; actual counts are displayed. No new
feature extraction is performed by the demo.

## Local launch

```powershell
.\.venv\Scripts\python.exe scripts\demo.py
```

Open `http://127.0.0.1:7860` on the PC. Default selections are InternVideo2 and All
datasets. If InternVideo2 features are absent, the first available model is used.
The initial encoder is warmed before launch; large checkpoint loading can take
time. Use `--no-warmup` to launch first and load on the first search instead.

Enter a description and press Enter or Search. Results are ranked by exact cosine
similarity. Select a thumbnail to prepare and play its video. ActivityNet playback
is trimmed to the retrieved interval. Previews are silent, matching the downloaded
datasets. Similarities are not confidence probabilities.

Model selection changes unload the previous encoder and warm the new model. Only
one encoder is retained on the GPU. GPU inference is serialized across sessions;
model switching takes longer than another search with the same model. Each
session has its own results/selection. Thumbnails and selected previews are
generated lazily, then cached for later requests.

## Laptop on another network

In the same PowerShell terminal on the PC:

```powershell
$env:VR_DEMO_USERNAME = "your-username"
$env:VR_DEMO_PASSWORD = "replace-with-your-password"
.\.venv\Scripts\python.exe scripts\demo.py --share
```

Open the printed `https://....gradio.live` URL on the laptop and log in with those
credentials. Localhost refers to the device running the browser, so the laptop
uses the printed HTTPS link rather than the PC's localhost URL. Sharing requires
both credential variables; they are not stored in source or configuration files.
The link is temporary and depends on Gradio's tunnel service. Keep the PC awake,
connected and the terminal running; create and test the link before presenting.
Ctrl+C stops the server. No router port forwarding is required for this mode.

The app serves generated media previews from its cache. Source raw-video
directories are blocked from direct file serving. Slower PC upload bandwidth can
delay playback; the result grid sends thumbnails and prepares only clicked videos.

## Custom paths and future datasets

```powershell
# Same built-in layouts under another base directory:
.\.venv\Scripts\python.exe scripts\demo.py --base-root "D:\video_retrieval"

# Explicit registry, with arbitrary future dataset keys:
.\.venv\Scripts\python.exe scripts\demo.py --config configs\demo.yaml
```

Use either `--base-root` or `--config`. Edit the supplied YAML roots to match your
machine. Relative dataset roots are resolved against the YAML file's directory;
relative feature paths are resolved against each dataset root.

```yaml
default_model: internvideo2
top_k: 10
datasets:
  extra_collection:
    label: Extra videos
    root: D:/extra_collection
    features:
      clip: features/clip_vit_b32_12f_mean
      clip4clip: features/clip4clip_vit_b32_12f_meanP
      internvideo2: features/internvideo2_stage2_1b_all
```

A registered dataset needs prepared compatible `features.npz`, `manifest.json`
and `run_config.json` files, plus source MP4 files under `raw_videos/`. Registering
a new dataset does not download videos or add a new annotation/extraction adapter.
Whole-video legacy bundles can use safe filename fallback; segment galleries
require complete source/interval metadata. Regenerate features after changing
source videos.

Combined search accepts only matching checkpoint provenance, tokenizer,
preprocessing, requested and effective extraction precision, and feature dimensions for the selected model. Full
checkpoint signatures (including stored paths) are compared conservatively. If
those settings differ, choose one dataset or extract matching galleries. Galleries
without recorded effective precision can be combined when all selected galleries
lack that field and their other saved settings match. The summary shows
"effective precision not recorded": historical dtype equality cannot be verified,
and query precision follows the saved requested setting. This supports older
bundles and runs that reused all cached features without loading an encoder.
Recorded and unknown effective precision are not combined. CUDA query
encoding must match recorded effective precision; explicit CPU mode uses fp32. IDs are
namespaced by dataset, so equal IDs in different collections stay distinct.

Other options: `--model clip`, `--top-k 10`, `--port 7861`, `--device cpu` (explicit
CPU inference), `--cache-dir PATH`. The default media cache is
`S:\video_retrieval\demo_cache`. It contains reproducible JPG thumbnails and
browser-compatible H.264 MP4 previews; stop the app before deleting it to reclaim
space. Do not put the preview cache inside `raw_videos/`. Preview generation has
timeouts; missing or unsupported media gets a placeholder/playback message.

## Presentation checks

Before presenting, test a known query, each model, combined and individual
datasets, and an ActivityNet segment. Repeat a query to confirm cached thumbnails.
Then open the share link over the laptop's other network and play a result. Keep
at least one model warmed before the presentation begins.

Local automated tests use synthetic galleries and CPU inference fixtures to
verify UI/search behavior, auth gating, file serving and real FFmpeg media.
The opt-in Playwright test passed for login, search, selectors and segment
delivery. The available test browser lacks H.264 decoding, so end-to-end
playback remains part of the PC/laptop presentation check. The
user's GPU, actual S: datasets and cross-network presentation must be tested on
their PC/laptop. The app does not implement composed retrieval or error-analysis
panels yet; those can reuse the separate query, search and media layers.
