# Shared video retrieval package

User-approved architecture: compare CLIP ViT-B/32 (12f mean), trained CLIP4Clip ViT-B/32 meanP (12f), and the working InternVideo2 Stage2 1B (4f), then reuse the same galleries for interaction.

Runtime data is external: S:\video_retrieval\msrvtt, S:\video_retrieval\vatex, S:\video_retrieval\activitynet. The repository data folder is reference-only; never auto-select it. Support explicit root/annotation overrides. Keep activitynet_captions as a dataset-name alias.

Use a standard src/visual_retrieval package with data, models, features, retrieval, interaction, and evaluation modules. Thin scripts provide preparation/extraction/evaluation/comparison/search commands. Model adapters expose encode_video RGB uint8 frames -> normalized[1,512], encode_text strings -> normalized[B,512], metadata and effective precision. Shared decoder samples actual intervals using timestamps; retain InternVideo2's corrected tokenizer and four-frame preprocessing.

Each dataset/model has its own provenance-checked output. Retain existing InternVideo2 cache schema/settings and old command entrypoints. Existing CLIP HDF5 tools remain legacy entrypoints; new comparisons use the shared NPZ format. Do not silently reinterpret legacy HDF5 caches.

CLIP4Clip requires an explicit trained checkpoint; validate full clip-prefixed ViT-B/32 weights, meanP/2d architecture and reject incompatible temporal heads. Use selected licensed upstream code, not random initialization or fallback CLIP weights. Shared evaluator verifies label ordering, protocol, completeness, and preprocessing. Comparison requires identical gallery IDs and query identities/text/labels across model outputs.

Interaction scope for this refactor: tested session history, accumulated/latest text queries, exact cosine search and relevance feedback utilities plus session evaluation. No UI, VQA, LLM or fine-tuning in this baseline milestone.

Validate old regression tests, CPU cached extraction/evaluation, lightweight real model architecture paths, CLI entrypoints from arbitrary cwd, external path defaults, mismatched comparisons and feedback edge cases. Full downloaded checkpoint inference needs user's GPU/videos and will be reported as unverified here.
