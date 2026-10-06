# InternVideo2 retrieval: MSR-VTT, VATEX, ActivityNet Captions

This program extracts video and text embeddings, evaluates text-to-video retrieval,
and searches videos with a new English query. No fine-tuning is performed.
Each dataset is evaluated separately against ONE gallery combining its selected
`train`, `test`, `val`, and `validation` lists (MSR-VTT retains train+test).
The `seed` and size/target metadata are ignored. There are no separate split scores.

## Model and implementation

Model: **OpenGVLab/InternVideo2-Stage2_1B-224p-f4**, full checkpoint
`InternVideo2-stage2_1b-224p-f4.pt` (about 2.82 GB).

Sources reviewed:
- https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2/multi_modality
- https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/multi_modality/demo/utils.py
- https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/multi_modality/demo/internvideo2_stage2_config.py
- https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/multi_modality/models/backbones/bert/tokenization_bert.py
- https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4

The `iv2_vendor` folder contains the selected official source files at commit
`3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3`, with attribution and license.
It avoids the main package's eager imports of unrelated models.
FlashAttention, fused MLP/RMSNorm, DeepSpeed, and custom CUDA compilation are
not required. PyTorch SDPA handles vision self-attention; native RMSNorm/MLP
and the official custom BERT implementation preserve the checkpoint's parameters.

The full Stage2 checkpoint loads the vision backbone, trained BERT, and **both
trained projection heads**. Missing weights used for retrieval raise an error.
The BERT model weights are not downloaded separately; only its tokenizer is.

Video processing uses 4 segment-center frames, RGB conversion, 224 x 224
uint8 bicubic resizing, and ImageNet normalization, following the official
evaluation dataset code. The model jointly encodes the four frames, then applies
its learned pooling and vision projection. It does not mean-pool independently
encoded frame embeddings. Text uses the official caption cleaning and maximum
40 tokens, the official **[CLS] + caption without a trailing [SEP]** format,
CLS features, and the trained text projection. Both produce normalized
512-dimensional vectors, and retrieval uses cosine similarity without VTM reranking.

## Expected data

```text
S:\video_retrieval\msr_vtt\
    subset.json
    raw_videos\video9605.mp4
    raw_videos\...
    raw_data\MSRVTT_data.json
```

Example `subset.json`:

```json
{"seed": 67, "train": ["video9605", "video1"], "test": ["video2"]}
```

The annotations must contain MSR-VTT's `sentences` list, for example:

```json
{"sentences": [{"video_id": "video9605", "caption": "A man plays the guitar.", "sen_id": 1}]}
```

This short annotation example is only illustrative. Your actual annotation file
must contain captions for every selected video. A file with video names alone
cannot provide retrieval ground truth. All caption rows are retained, including
multiple captions for the same video. Captions for unselected videos are ignored.
If your annotations live elsewhere, pass `--annotations "FULL_PATH"`.

## Windows PowerShell installation

Clone `KarenYuusha/visual-retrieval` and open its `internvideo2_retrieval` folder. These commands
use `uv` and a fresh Python 3.11 environment. Activation is not required.

```powershell
cd S:\video_retrieval\visual-retrieval\internvideo2_retrieval
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -r requirements.txt
```

If you prefer pip and already have Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements explicitly pin `torch==2.5.1+cu121` and
`torchvision==0.20.1+cu121`, so a CPU build cannot satisfy them. The uv command
uses pip-style index selection because the PyTorch index can also contain
different versions of shared dependencies. CUDA 12.1 wheels support Linux and
Windows; these GPU requirements are not intended for macOS.

If you previously installed a CPU build, run the updated installation command
above in the same environment. Verify the result:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Expected: `2.5.1+cu121`, `12.1`, and `True`. A CUDA build with `False` means
PyTorch still cannot access your GPU/driver; it is not a CPU-only package.

A CUDA-capable NVIDIA GPU is recommended. Video batch size is 1, text batch
size defaults to 32, and precision defaults to bf16 on supported GPUs or fp16
otherwise. Actual VRAM use depends on the GPU and PyTorch backend. Allow space
for the model cache and sufficient host RAM for checkpoint loading. CPU extraction
is supported in fp32 but is slow.

## Run all three datasets

Default dataset roots are `S:\video_retrieval\msr_vtt`,
`S:\video_retrieval\vatex`, and `S:\video_retrieval\activitynet_captions`.
All require `subset.json` and `raw_videos/<video ID>.mp4`.

| Dataset | Annotations in `raw_data/` (or dataset root) | Gallery and queries |
|---|---|---|
| MSR-VTT | `MSRVTT_data.json` | Whole videos, every caption independently |
| VATEX | `vatex_training_v1.0.json`, `vatex_validation_v1.0.json` | Downloaded clips, every English `enCap` caption; no Chinese queries |
| ActivityNet Captions | `train.json`, `val_1.json` | Default: timestamped segments, one sentence per segment |

```powershell
# Preflight all datasets; no GPU/checkpoint download or video decoding
.\.venv\Scripts\python.exe run_datasets.py --stage check

# Extract then evaluate all three, sequentially on the GPU
.\.venv\Scripts\python.exe run_datasets.py

# Reevaluate saved features without a GPU/model
.\.venv\Scripts\python.exe run_datasets.py --stage evaluate
```

For data inside the cloned repository use `--base-root "..\data"`.
For independently located datasets, use `--msrvtt-root`, `--vatex-root`, and
`--activitynet-captions-root`. `--datasets vatex activitynet_captions` selects
only those datasets. `--checkpoint "S:\video_retrieval\models\InternVideo2-stage2_1b-224p-f4.pt"`
reuses a local checkpoint. `--text-batch-size 8` reduces text inference memory.
The runner loads one dataset/model at a time and reuses downloaded weights.

Outputs go into **each dataset's** `features/internvideo2_stage2_1b_all/`.
An optional `--output-root "S:\video_retrieval\iv2_features"` instead produces
`msrvtt/`, `vatex/`, and `activitynet_captions/` subfolders there.
The summary is saved by default to
`S:\video_retrieval\features\internvideo2_summary.json` (or under `--base-root`);
use `--summary PATH` to change it. It contains separate results for each dataset.

### ActivityNet protocol

Default `--activitynet-mode segments` matches this repository's CLIP segment
extraction: each `v_ID__seg0000` gallery item represents its own `[start,end)`
interval in the original MP4. Frames are sampled only inside that interval using decoded presentation timestamps
(including variable-frame-rate videos);
segments outside the decoded video fail explicitly. A timestamp end beyond
actual duration uses only available frames; a short interval repeats its final available frame.
Several segments can share a source video. Correctness means retrieving the
**labeled segment**, not any other segment from the same source.

For whole-video retrieval, use `--activitynet-mode video`: one video embedding
and one paragraph query formed by joining all its sentences. Text still uses
40 tokens, so long paragraphs are truncated. These scores represent a different
protocol. Use a separate `--output-root` when switching modes. Validation uses
`val_1.json`; `--activitynet-val-file val_2.json` explicitly selects the alternate
annotations. Both validation versions are never silently merged.

### Run one dataset directly

```powershell
.\.venv\Scripts\python.exe extract_features.py --dataset vatex
.\.venv\Scripts\python.exe evaluate_retrieval.py --features-dir "S:\video_retrieval\vatex\features\internvideo2_stage2_1b_all"

.\.venv\Scripts\python.exe extract_features.py --dataset activitynet_captions
.\.venv\Scripts\python.exe evaluate_retrieval.py --features-dir "S:\video_retrieval\activitynet_captions\features\internvideo2_stage2_1b_all"
```

`extract_features.py` still defaults to MSR-VTT, preserving the previous command
and cache configuration. Use `--annotations PATH [PATH ...]` for explicit
annotation files (one file for MSR-VTT). `--check-data` checks annotations and
file existence but does not decode videos. Missing captions are errors.
An incomplete extraction exits with an error after saving its partial bundle,
and stops the all-dataset runner. Retry extraction to complete it; if deliberately
scoring a partial gallery, run evaluation separately with `--allow-incomplete`
(or `run_datasets.py --stage evaluate --allow-incomplete`).

## Updating from the first version: tokenizer correction

The first version used the standard BERT tokenizer, which added a trailing
`[SEP]`. The official InternVideo2 tokenizer deliberately omits `[SEP]` for
single captions. That mismatch changes both token IDs and attention masks.
This version restores the official format using a compatible tokenizer subclass.

Replace the program files with this version, keeping your dataset/features
folder intact, then rerun **the same extraction command you used before**:

```powershell
.\.venv\Scripts\python.exe extract_features.py
.\.venv\Scripts\python.exe evaluate_retrieval.py
```

If you previously supplied `--checkpoint`, `--precision`, `--annotations`,
`--output-dir`, or other settings, use those same settings. When the tokenizer
version is the only change, the migration retains `video_cache`, deletes old
text/combined feature caches, and recomputes text embeddings. Old evaluation
results are retained as `evaluation_before_tokenizer_fix.json` and
`query_ranks_before_tokenizer_fix.csv`. It does not re-encode unchanged videos.
Other settings or checkpoint changes still require a fresh output directory.

The corrected tokenizer does not establish what your new scores will be;
evaluate again after regenerating text embeddings.

## 2. Extract features

```powershell
.\.venv\Scripts\python.exe extract_features.py
```

The first run automatically downloads the exact checkpoint filename and the
BERT tokenizer. If Hugging Face requests access, open the model page, accept its
conditions using your account, and authenticate:

```powershell
.\.venv\Scripts\huggingface-cli.exe login
```

Then rerun extraction. If you already downloaded the correct full checkpoint:

```powershell
.\.venv\Scripts\python.exe extract_features.py --checkpoint "S:\video_retrieval\models\InternVideo2-stage2_1b-224p-f4.pt"
```

To use a different dataset location and annotation file:

```powershell
.\.venv\Scripts\python.exe extract_features.py --data-root "D:\datasets\msr_vtt" --annotations "D:\datasets\MSRVTT_data.json"
```

The default output folder is:

```text
S:\video_retrieval\msr_vtt\features\internvideo2_stage2_1b_all
```

Re-running the same extraction command reuses completed video caches and retries
decode failures. Video size/mtime changes invalidate that video's cached embedding.
Settings, subset, annotation, or checkpoint changes require a new `--output-dir`
to prevent mixing incompatible embeddings.
Truncated cache files are rebuilt. Existing caches without `run_config.json`
are rejected because their model/settings provenance cannot be verified.

Missing files stop extraction before downloading a model. `--allow-missing`
explicitly permits extracting the remaining files. Decode failures are logged
and extraction continues. Model-loading/inference errors abort immediately, keeping
already completed caches. An incomplete extraction saves its available features
and exits with an error; evaluation also refuses incomplete galleries by default.

## 3. Evaluate

```powershell
.\.venv\Scripts\python.exe evaluate_retrieval.py
```

Evaluation loads only saved features; it needs no model, GPU, or video decoding.
Every caption is a query, its labeled video is the single correct gallery video,
and all combined train+test videos are candidates for every query. Around 2,000
videos with 20 captions per video would yield around 40,000 queries; the program
uses your actual counts.

- R@1, R@5, R@10: percentage of captions whose correct video appears in the top k.
- MRR: mean of `1 / correct_video_rank`, using the **full gallery ranking**,
  reported on a 0–1 scale. It is not truncated at rank 10.
- Ties: deterministic gallery order, first occurrence in train+test.
- Averaging: each caption has equal weight.

Scores are printed and saved to `evaluation.json`; individual ranks are saved
to `query_ranks.csv`. The JSON contains gallery/query counts, completeness,
omissions, metric units, and the protocol. It reports only combined-gallery
text-to-video metrics, as requested.

For a custom extraction output, pass the same directory:

```powershell
.\.venv\Scripts\python.exe evaluate_retrieval.py --features-dir "D:\features\iv2"
```

If you deliberately want an evaluation on successfully extracted videos despite
failures, add `--allow-incomplete`. The report labels that run as incomplete and
lists omissions. This produces a different gallery size.

These are project-specific combined-subset scores. Their gallery and all-caption
protocol differ from standard MSR-VTT 1K test results; compare other models using
the same gallery, queries, and ranking definitions.

## 4. Search

```powershell
.\.venv\Scripts\python.exe search_video.py --query "a man carrying an umbrella" --top-k 10
```

Only the query is newly encoded. The command uses the saved video embeddings
and prints video IDs, cosine scores, and local file paths. Optional `--output`
saves a JSON result file. A custom feature location requires `--features-dir`.

## Files produced

| File | Contents |
|---|---|
| `features.npz` | `video_features [N,512]`, `video_ids [N]`, `text_features [Q,512]`, `text_video_ids [Q]`; float32, no pickle |
| `queries.json` | Caption text, annotation query ID, and correct video ID in feature-row order |
| `manifest.json` | Requested/successful IDs, failures, settings, and provenance |
| `run_config.json` | Cache compatibility settings |
| `video_cache/*.npz` | Individual resumable video embeddings, sample indices, source signatures |
| `text_cache.npz` | All selected-caption embeddings for reuse |
| `failed_videos.json` | Missing or undecodable videos |
| `evaluation.json` | Combined-gallery R@1/R@5/R@10/MRR |
| `query_ranks.csv` | Rank and reciprocal rank for every evaluated caption |

## Validation and limitations

Install development requirements and run the suite:

```powershell
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The suite checks ID merging, caption alignment, exact ranking/metrics, ties,
the evaluation CLI, all-dataset cached extraction, temporal interval sampling,
real short-video decoding, preprocessing, checkpoint safety,
and native attention equivalence. The wrapper is exercised against reduced-size
instances of the actual official vision and BERT architectures.

The full 1B checkpoint inference and your S: drive videos have not been executed
in the development environment. No real dataset metrics are supplied or fabricated.
