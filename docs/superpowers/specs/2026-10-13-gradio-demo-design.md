# Archive: Gradio demo design (2026-10-13)

Historical requirements; the UI has since gained composed search and pagination.
Current usage: [demo](../../demo.md).

- GPU PC serves text retrieval to a remote laptop through authenticated opt-in sharing.
- Combine only matching model spaces; namespace dataset IDs and show partial gallery counts.
- One live GPU encoder; serialize inference and release before switching models.
- Lazy FFmpeg cache; preserve ActivityNet intervals and serve generated previews only.
- Configurable dataset roots; retain CUDA pins and checkpoint provenance checks.
- Validate search, model lifecycle, media generation, stale events and authentication.
