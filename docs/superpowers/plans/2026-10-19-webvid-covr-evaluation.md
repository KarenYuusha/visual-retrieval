# Archive: WebVid-CoVR evaluation implementation (2026-10-19)

Historical plan summary. [Design](../specs/2026-10-19-webvid-covr-evaluation.md) ·
[Current commands](../../webvid_covr_evaluation.md).

- Added subset adapter, baseline extraction, composed-query caches and sequential CLI.
- Encodes four model spaces; evaluates only CoVR-2 with full-gallery ranks.
- Preserves stable IDs, source/image/text provenance, reference exclusion and explicit omissions.
- Enforces reference-image identity and separate requested/downloaded/evaluated counts.
- Initial verification: 162 passed, three opt-in browser tests skipped; real-media fixture
  confirmed cache-only resume and one live encoder. Later throughput changes extend this work.
