# Optional multi-turn experiments

Backend-only: `latest`, `accumulated`, or `feedback` query methods.
For the main image + text workflow, use [CoVR-2](covr2.md).

Create a sessions JSON file; replace these IDs with real gallery IDs:

```json
[
  {"session_id": "s1", "target_item_id": "video9605", "turns": [
    {"query": "a man outdoors"},
    {"query": "carrying an umbrella"}
  ]}
]
```

Run from `P:\visual-retrieval`:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_sessions.py `
  --features-dir "S:\video_retrieval\msr_vtt\features\internvideo2_stage2_1b_all" `
  --sessions "S:\video_retrieval\sessions\msrvtt_test.json" `
  --method accumulated --output "S:\video_retrieval\reports\iv2_sessions.json"
```

Reports per-turn R@1/5/10, MRR, ranks, latency and first top-10 turn.
Feedback turns may include `positive`/`negative` ID lists, applied to the next
turn as `normalize(q + 0.5 * mean(positive) - 0.25 * mean(negative))`.
Use matched held-out sessions; label automatic ground-truth feedback as oracle simulation.
