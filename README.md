# Visual retrieval

Video retrieval experiments on downloaded MSR-VTT, VATEX, and ActivityNet Captions subsets.

The complete **InternVideo2 Stage2 1B** pipeline is in [internvideo2_retrieval](internvideo2_retrieval/README.md).
It includes CUDA installation, feature extraction with resumable caches, text-to-video R@1/R@5/R@10/MRR evaluation, and text search.

```powershell
cd internvideo2_retrieval
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -r requirements.txt
.\.venv\Scripts\python.exe run_datasets.py --stage check
.\.venv\Scripts\python.exe run_datasets.py
```

The default data location is `S:\video_retrieval`, with `msr_vtt`, `vatex`, and
`activitynet_captions` subfolders. Use `--base-root "..\data"` if videos are stored
alongside this repository's annotations. See the pipeline README for all paths,
protocols, individual commands, and saved file formats.

Each dataset uses its combined selected splits as one gallery. ActivityNet defaults
to segment retrieval; whole-video paragraph retrieval is an explicit alternative.
These subset results are distinct from official benchmark results.

Existing CLIP extraction/evaluation scripts and download scripts remain available at
the repository root and under `download_data/`.
