# CoVR-2 composed search

Approved scope: one reference image plus modification text retrieves ranked videos
using the released CoVR-2 coir+covr checkpoint. Text alone continues to use the
existing models. The single composer accepts image upload, drag/drop and paste;
the current displayed video frame can be copied as PNG or attached directly.
Use a fixed composed top 100, paginated 20 per page, without a Top K UI control.
Keep the responsive borderless gallery, dataset filter and authenticated sharing.

Preserve the upstream BLIP-2 COCO target encoder, trained composed Q-Former,
per-token normalization, token averaging, and text-dependent frame weighting.
Use 15 uniformly sampled target frames, within ActivityNet segment boundaries.
Create separate resumable feature caches with checkpoint/preprocessing provenance.
Do not use InternVideo2 embeddings or a text-only candidate prefilter for CoVR-2.
Only one retrieval engine is active on the GPU. Validate actual memory rather
than promise that both engines fit the user's 12 GB GPU simultaneously.

Copy means the paused/current displayed frame, without browser controls. Pasting
preserves entered text. Allow one decoded raster image; reject video/multiple
images. Capture known source metadata for direct attachment and exclude the known
reference item from composed results; external images have no assumed source ID.
Validate stale events, model/dataset changes, copied segment timing, browser
clipboard permission failure, and numerical parity with the upstream pipeline.
