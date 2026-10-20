# Inspect retrieval errors

## Export predictions

Run from `P:\visual-retrieval`; replace the feature path with your model/dataset:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_baselines.py `
  --features-dir "S:\video_retrieval\msr_vtt\features\clip_vit_b32_12f_mean" --top-k 10
```

Uses saved embeddings on CPU. Add `--allow-incomplete` only for reviewed partial
bundles. `--top-k` changes exported candidates, never full-gallery metrics.

| File | Use |
|---|---|
| `evaluation.json` | Metrics, protocol, omitted IDs |
| `query_ranks.csv` | Filter wrong top-1 results and inspect target ranks/scores |
| `predictions.jsonl` | Query, labels, rank and top-k candidate IDs/scores |
| `manifest.json` | Map IDs to source paths and segment boundaries |

`ground_truth_video_id` is the target; `predicted_video_id` is top-1.
`is_correct` requires an exact item-ID match, including ActivityNet segment ID.
Baseline `score_gap` is predicted minus target cosine score, not confidence.
Ties follow gallery order. Legacy bundles may lack path/interval metadata.

Compare models on identical queries/galleries. Review wrong object/action,
missed temporal detail, ambiguous captions, near duplicates and decoding errors.
Keep manual tags separate from ground truth. A semantically suitable alternative
video still counts as an ID mismatch. A dedicated error-analysis viewer is not implemented.
