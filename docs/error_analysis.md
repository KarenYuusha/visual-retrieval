# Prediction exports and visual error analysis

Evaluation uses caption queries as inputs and gallery video/segment IDs as labels.
The correct label is the dataset's ground-truth ID, not an automatically corrected
caption. Top-1 is correct only when its item ID matches the ground truth; a different
ActivityNet segment from the same source video is still an ID mismatch.

## Files

Each evaluated feature directory contains:

| File | Contents |
|---|---|
| `evaluation.json` | Metrics, protocol, tie policy, omitted IDs and prediction-export schema/version |
| `query_ranks.csv` | Query/caption, full-gallery target rank, ground-truth/top-1 IDs, correctness, scores, paths and segment times |
| `predictions.jsonl` | One JSON object per query with labels, rank, scores, ground-truth/top-1 metadata and top-k IDs/scores |
| `manifest.json` | Gallery item metadata including paths, source video IDs and segment intervals |

The original CSV columns `query_id`, `video_id`, `caption`, `rank` and
`reciprocal_rank` are retained. `video_id` and `ground_truth_video_id` both refer to
the target. `predicted_video_id` is the highest-scoring item. `is_correct` is an
exact item-ID match. CSV uses `True`/`False`, while JSON uses booleans. Saved
`predicted_score` and `ground_truth_score` are cosine similarities; `score_gap` is
their difference, not confidence or a probability. Wrong top-1 results with a zero
gap can occur because ties prefer the earlier gallery item.

JSON `top_k` candidates have `rank`, `video_id`, `score`, and `is_ground_truth`.
Their metadata can be looked up by `item_id` in the saved manifest. Ground-truth
and top-1 metadata include `item_id`, `source_video_id`, `path`, `start`, and `end`.
Legacy bundles without gallery metadata have null item metadata and blank CSV path
fields; filenames/timestamps are not inferred. For whole videos, segment times are
null. Keep the full saved caption for display; its model-specific tokenization and
truncation are part of the declared preprocessing.

```powershell
cd P:\visual-retrieval
.\.venv\Scripts\python.exe scripts\evaluate_baselines.py --features-dir "S:\video_retrieval\activitynet_captions\features\clip4clip_vit_b32_12f_meanP" --allow-incomplete --top-k 10
```

Reevaluation uses saved embeddings on the CPU and does not load encoders or decode
videos. Increasing `--top-k` increases export size; the JSONL writer streams rows
and scoring remains batched. Full-gallery ranks and MRR are unchanged. Under
`--allow-incomplete`, only queries present in the feature bundle are evaluated;
omitted gallery IDs remain in `evaluation.json`. New extraction invalidates the
prediction files so old labels cannot silently describe a new feature bundle.

## Proposed local viewer

The following is a suggested next step, not an implemented UI.

1. Show the query caption at the top. Display the ground-truth video and predicted
   top-1 side by side, with IDs, scores and the target rank. For ActivityNet, play
   or preview the exact annotated intervals; different segments must not appear
   as indistinguishable full-video players.
2. Show a gallery of top-10 result thumbnails or small storyboards, with rank and
   similarity. Highlight the ground-truth item if it appears. Selecting a card
   opens its source video at the segment start. Use manifest metadata for lookup.
3. Filter cases by model, dataset, correctness, target-rank bucket (1, 2–5, 6–10,
   >10), score gap and caption search. A table of wrong top-1 cases provides a
   simple starting point before building video playback.
4. Compare CLIP, CLIP4Clip and InternVideo2 for the same query. Match queries using
   `(query_id, ground_truth_video_id, caption)` and verify the same gallery and
   segment intervals. Mark queries absent from one model rather than pretending
   they were retrieved incorrectly. Compare model-specific score gaps cautiously;
   their similarity distributions need not have the same calibration.
5. Save manual error tags separately from the dataset ground truth: wrong object,
   wrong action, missed temporal detail, near-duplicate video, ambiguous caption,
   possible annotation issue, or decoding issue. These tags support offline
   analysis and do not change retrieval labels or feed back into the model.

Useful project figures include correct-at-1/5/10 counts, target-rank distributions,
win/loss counts for pairs of models on matched queries, and manually tagged error
counts. ID mismatches are not automatically semantic failures: another video may
also satisfy a generic caption. Review those cases visually and preserve the
original benchmark labels when reporting R@K/MRR.
