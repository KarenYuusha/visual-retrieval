# Interactive experiments

Three supported methods: latest query, accumulated query, accumulated query plus relevance feedback. Feedback is applied within the active model's embedding space and affects the next turn. A feedback label can be reversed; session.reset() clears history/feedback.

Update: normalize(q + beta * mean(positive vectors) - gamma * mean(negative vectors)). Defaults beta0.5/gamma0.25; tune on development sessions only. Zero/invalid vectors and incompatible dimensions are errors.

Example sessions JSON (illustrative IDs; replace with actual gallery items):

```json
[
  {
    "session_id": "session001",
    "target_item_id": "video9605",
    "turns": [
      {"query": "A man outdoors", "positive": ["video9605"], "negative": ["video1"]},
      {"query": "He is carrying an umbrella"}
    ]
  }
]
```

Positive/negative item IDs represent recorded feedback after that turn. Omitting them yields text refinement only. Do not automatically select the ground-truth target as a positive without labeling the experiment as simulated oracle feedback. Separate development/test sessions by source video, especially for ActivityNet segments.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_sessions.py --features-dir "S:\video_retrieval\msrvtt\features\internvideo2_stage2_1b_all" --sessions "S:\video_retrieval\sessions\msrvtt_test.json" --method accumulated --output "S:\video_retrieval\reports\iv2_sessions.json"
```

Use --method latest or feedback for alternatives. The report includes per-turn R@1/R@5/R@10/MRR, raw ranks, latency and first turn reaching top10. Turn metrics include sessions that have that turn and report the denominator; use equally sized turn sequences for matched comparisons. End-to-end latency includes text encoding, feedback and ranking; GPU-to-CPU feature transfer synchronizes inference.

Build a manually checked small session set first. Natural multi-turn descriptions should reveal additional target details gradually. Match sessions across models/methods; report failures that remain unretrieved, rather than averaging only successful sessions.
