# VATEX Subset

Seed: 67

Train target: 1000
Train successful: 1000

Validation target: 1000
Validation successful: 1000

Download strategy:
Direct download only (no availability pre-scan)

Workers: 3
Encode concurrency: 2

YouTube JS runtime: deno
Browser cookies: firefox

Source resolution: <= 360p
Audio: disabled

Encoder: libx264

yt-dlp process timeout: 300s
Rate-limit backoff: 90s

Source cache limit: 2.0 GB

subset.json contains only successfully downloaded and
ffprobe-validated clips.

Old dead_sources.json entries are not trusted unless they have
verified_with_auth=true. This allows sources classified while the
old yt-dlp environment was broken to be tried again.
