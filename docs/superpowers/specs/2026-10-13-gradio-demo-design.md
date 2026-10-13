# Minimal Gradio video-search demo

Approved scope: run text-to-video retrieval on the user's Windows GPU PC and open
the password-protected temporary Gradio link from a laptop on another network.
Use CLIP, CLIP4Clip and InternVideo2; default to InternVideo2 and all available
datasets. Keep the page minimal: title, query/Search, compact model/dataset
selectors, top-10 thumbnails, and one selected-result player.

Search reuses saved video embeddings and the matching existing encoder. Combine
only galleries with matching checkpoint signatures, tokenizer, preprocessing,
source implementation and precision. Namespace item IDs by dataset. Load one
encoder at a time; release it before loading a replacement, serialize GPU work,
and retain video embeddings on the CPU. Partial galleries are searchable and
their actual item counts are displayed. Missing or invalid galleries are reported
at startup and unavailable choices excluded. Combined incompatibility produces a
clear error; a single compatible gallery remains searchable.

Dataset locations are configurable and independent of hard-coded adapter names:
future prepared datasets can be registered with a label, data root, and per-model
feature folders. Resolve source videos under the configured raw_videos directory;
require segment metadata for ActivityNet. Validate names, dimensions and time
intervals rather than guessing segment bounds.

Media is generated lazily with FFmpeg. Cache fingerprints include file path,
size, modification time, interval and rendering version. Thumbnails depict the
segment midpoint (or whole-video midpoint); selected ActivityNet previews are
trimmed to the actual interval and encoded as browser-compatible H.264 MP4.
Previews are silent, matching the downloaded datasets. Atomic writes and per-key
locks prevent partial cached files. Missing/invalid media yields a placeholder or
clear playback message rather than hiding search results. Serve only the media
cache directory through the UI, not arbitrary model/data roots.

Use Gradio 6.9.0 and huggingface-hub 0.36.2, compatible with transformers 4.40.2.
Keep torch 2.5.1+cu121 and torchvision 0.20.1+cu121 pinned. Add an optional demo
dependency extra and requirements-demo.txt. Sharing requires both username and
password environment variables; no real credentials are stored in the repo.
Local launch binds 127.0.0.1 and does not share unless explicitly requested. The
PC must remain running. Actual cross-network/GPU presentation verification is
performed on the user's hardware, not claimed from local CPU fixture tests.

Separate query encoding, gallery retrieval and presentation. The current text-only
search interface accepts text/model/dataset selection; future image+text composition
can replace query encoding without restructuring ranking or media components.
No composed-retrieval implementation or error-analysis dashboard is included now.

Acceptance: search ranks match the existing cosine index; combined datasets and
colliding IDs behave correctly; mismatched spaces fail; model lifecycle reuses
and releases encoders; missing bundles/media and malformed intervals are handled;
real FFmpeg previews preserve intervals; the Gradio page passes local browser
interaction tests; auth gates remote launch; documented Windows install keeps GPU
pins and includes launch, sharing, custom dataset configuration and cache usage.
