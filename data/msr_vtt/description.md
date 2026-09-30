### MSR-VTT Dataset Subset

This project uses a reduced subset of the **MSR-VTT (Microsoft Research Video to Text)** dataset for video retrieval experiments.

The original MSR-VTT dataset contains approximately **10,000 videos** and **200,000 natural-language captions**, with around **20 captions describing each video**.

Due to storage and computational constraints, the current subset contains:

* **Training set:** 1,000 videos sampled from the MSR-VTT training split
* **Test set:** 1,000 videos from the standard MSR-VTT 1K test split
* **Captions:** All available captions associated with each selected video are retained
* **Approximate training text-video pairs:** 20,000
* **Approximate test text-video pairs:** 20,000

Each video is identified by a unique `video_id`, such as `video123`. Captions are stored separately in `MSRVTT_data.json` under the `sentences` field and are linked to videos through the corresponding `video_id`.

The dataset therefore follows a many-to-one relationship:

`multiple text captions → one video`

For example, a single video may have approximately 20 different textual descriptions referring to the same visual content.

The dataset is intended primarily for **text-to-video retrieval** experiments. A text query is encoded into a feature representation and compared against representations of candidate videos. Videos are ranked according to their similarity to the query.

Typical evaluation metrics include:

* Recall@1 (R@1)
* Recall@5 (R@5)
* Recall@10 (R@10)
* Median Rank (MedR)
* Mean Rank (MnR)

The 1,000-video training set is a reduced subset used to lower storage requirements and speed up development and experimentation. The 1,000-video test set can be used as the retrieval gallery during evaluation.

When training, all captions belonging to a selected training video should be retained rather than using only one caption per video. This provides multiple textual descriptions of the same visual content and produces approximately 20 training pairs per video.

Example data relationship:

```text
video123.mp4
    ├── "a man is playing a guitar"
    ├── "a person performs music with a guitar"
    ├── "a man plays an instrument"
    ├── ...
    └── approximately 20 captions
```

Current dataset organization can be considered conceptually as:

```text
MSR-VTT
├── train
│   └── 1,000 videos
│
├── test
│   └── 1,000 videos
│
└── MSRVTT_data.json
    ├── videos
    └── sentences
         ├── video_id
         └── caption
```

For reproducibility, the training subset should be selected using a fixed random seed and the selected `video_id` list should be saved separately.
