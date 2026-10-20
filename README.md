# Interactive video retrieval

Search videos with text (CLIP, CLIP4Clip, InternVideo2) or a reference image +
text modification (CoVR-2). Uses pretrained weights; no local training.

## Start here

Run commands in **PowerShell from `P:\visual-retrieval`**. Dataset files stay under
`S:\video_retrieval`; the repository's `data/` folder is reference-only.

| I want to… | Guide |
|---|---|
| Extract/evaluate CLIP, CLIP4Clip, InternVideo2 | [Text retrieval baselines](docs/baselines.md) |
| Open the demo locally or from my laptop | [Demo](docs/demo.md) |
| Search using an image + text | [CoVR-2 setup](docs/covr2.md) |
| Download 5 WebVid-CoVR samples | [WebVid-CoVR download](docs/webvid_covr_download.md) |
| Encode WebVid-CoVR and evaluate only CoVR-2 | [WebVid-CoVR evaluation](docs/webvid_covr_evaluation.md) |
| Find dataset/checkpoint/output details | [Datasets](docs/datasets.md) · [Models](docs/models.md) · [Error analysis](docs/error_analysis.md) |

## Install once

Requires Python 3.11/3.12, an NVIDIA GPU, and FFmpeg/ffprobe on PATH.
For a new environment:

```powershell
cd P:\visual-retrieval
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -r requirements-demo.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
ffmpeg -version
ffprobe -version
```

Expected Torch output: `2.5.1+cu121 12.1 True`. Use the requirements files to keep
CUDA builds pinned. For an existing environment, skip `uv venv`; run `git pull`
and the install command when dependencies change.

## Develop

```powershell
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

[Code map](docs/project_structure.md) · [Optional session experiments](docs/interaction.md).
Historical plans/specs in `docs/superpowers/` are archive notes, not run instructions.
