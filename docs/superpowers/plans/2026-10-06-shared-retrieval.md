# Archive: shared retrieval implementation (2026-10-06)

Historical plan summary. [Design](../specs/2026-10-06-shared-retrieval-design.md) ·
[Current commands](../../baselines.md).

- Migrated shared code to `src/visual_retrieval`; retained compatibility wrappers.
- Added three encoder adapters, strict trained-checkpoint loading and atomic caches.
- Added matched comparisons, exact search and optional multi-turn/feedback utilities.
- Required identical gallery/query identities for comparisons and explicit external roots.
- Verified legacy regressions, reduced model paths, cached extraction and CLI routing.
