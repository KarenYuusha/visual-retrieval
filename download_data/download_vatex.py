import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("vatex")

# These are SUCCESSFUL final clip targets.
TRAIN_SIZE = 1000
VAL_SIZE = 1000

SEED = 67

# Actual media-download workers. 3 is a safer default when using
# an authenticated YouTube session; increase to 4 only if stable.
WORKERS = 3

# Limit simultaneous FFmpeg encodes.
ENCODE_CONCURRENCY = 2

# Download video-only source streams at <= 360p.
MAX_HEIGHT = 360

# yt-dlp / YouTube environment that you verified works.
JS_RUNTIME = "deno"
COOKIES_FROM_BROWSER = "firefox"

# Hard timeout for one source-video yt-dlp process.
YTDLP_PROCESS_TIMEOUT = 300
YTDLP_RETRIES = 2
YTDLP_FRAGMENT_RETRIES = 2
YTDLP_EXTRACTOR_RETRIES = 1
SOCKET_TIMEOUT = 20
THROTTLED_RATE = "100K"

# If YouTube responds with 429 / bot challenge, pause before
# launching more source requests instead of hammering it.
RATE_LIMIT_BACKOFF_SECONDS = 90

# Full source cache. Multiple VATEX clips from the same YouTube
# video can reuse one source download.
SOURCE_CACHE_MAX_GB = 2.0
CLEAR_SOURCE_CACHE_AT_END = True

SKIP_EXISTING = True

# Old failed.json entries were produced while the yt-dlp
# environment was unreliable. Retry them now.
RETRY_FAILED = True

# IMPORTANT: old dead_sources.json may contain false positives
# from the broken environment. Only entries explicitly marked
# verified_with_auth=true are trusted/skipped.
TRUST_ONLY_AUTH_VERIFIED_DEAD = True

# GPU encoding if FFmpeg + NVIDIA driver support it.
NVENC_PRESET = "p3"
NVENC_CQ = 24

# CPU fallback.
X264_PRESET = "veryfast"
X264_CRF = 23

# Windows-safe JSON checkpoint writes.
JSON_SAVE_RETRIES = 20
JSON_RETRY_DELAY = 0.10


# ============================================================
# PATHS
# ============================================================

RAW_DATA_DIR = ROOT / "raw_data"
VIDEO_DIR = ROOT / "raw_videos"
CACHE_DIR = ROOT / "source_cache"

SUBSET_FILE = ROOT / "subset.json"
FAILED_FILE = ROOT / "failed.json"
DEAD_FILE = ROOT / "dead_sources.json"
DESCRIPTION_FILE = ROOT / "description.md"

TRAIN_JSON = RAW_DATA_DIR / "vatex_training_v1.0.json"
VAL_JSON = RAW_DATA_DIR / "vatex_validation_v1.0.json"

for directory in (RAW_DATA_DIR, VIDEO_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TRAIN_URL = (
    "https://eric-xw.github.io/vatex-website/data/"
    "vatex_training_v1.0.json"
)

VAL_URL = (
    "https://eric-xw.github.io/vatex-website/data/"
    "vatex_validation_v1.0.json"
)


# ============================================================
# SHARED THREAD STATE
# ============================================================

JSON_SAVE_LOCK = threading.Lock()
SOURCE_LOCKS_GUARD = threading.Lock()
ACTIVE_SOURCE_LOCK = threading.Lock()
RATE_LIMIT_LOCK = threading.Lock()

SOURCE_LOCKS = {}
ACTIVE_SOURCES = {}
RATE_LIMIT_UNTIL = 0.0

ENCODE_SEMAPHORE = threading.Semaphore(ENCODE_CONCURRENCY)
VIDEO_ENCODER = "libx264"


def get_source_lock(video_id):
    with SOURCE_LOCKS_GUARD:
        lock = SOURCE_LOCKS.get(video_id)
        if lock is None:
            lock = threading.Lock()
            SOURCE_LOCKS[video_id] = lock
        return lock


def acquire_source_use(path):
    key = str(Path(path).resolve())
    with ACTIVE_SOURCE_LOCK:
        ACTIVE_SOURCES[key] = ACTIVE_SOURCES.get(key, 0) + 1


def release_source_use(path):
    key = str(Path(path).resolve())
    with ACTIVE_SOURCE_LOCK:
        count = ACTIVE_SOURCES.get(key, 0)
        if count <= 1:
            ACTIVE_SOURCES.pop(key, None)
        else:
            ACTIVE_SOURCES[key] = count - 1


def source_is_active(path):
    key = str(Path(path).resolve())
    with ACTIVE_SOURCE_LOCK:
        return ACTIVE_SOURCES.get(key, 0) > 0


def trigger_rate_limit_backoff():
    global RATE_LIMIT_UNTIL
    with RATE_LIMIT_LOCK:
        RATE_LIMIT_UNTIL = max(
            RATE_LIMIT_UNTIL,
            time.time() + RATE_LIMIT_BACKOFF_SECONDS,
        )


def wait_for_rate_limit_backoff():
    while True:
        with RATE_LIMIT_LOCK:
            remaining = RATE_LIMIT_UNTIL - time.time()

        if remaining <= 0:
            return

        time.sleep(min(5.0, remaining))


# ============================================================
# SAFE JSON I/O
# ============================================================


def save_json(data, path):
    """Windows-safe checkpoint writer that never kills the run."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with JSON_SAVE_LOCK:
        temp_path = None

        try:
            fd, temp_name = tempfile.mkstemp(
                prefix=path.name + ".",
                suffix=".tmp",
                dir=str(path.parent),
            )
            temp_path = Path(temp_name)

            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())

            for attempt in range(JSON_SAVE_RETRIES):
                try:
                    os.replace(temp_path, path)
                    return True
                except OSError as e:
                    if attempt == JSON_SAVE_RETRIES - 1:
                        print(
                            f"\nWARNING: could not save {path} after "
                            f"{JSON_SAVE_RETRIES} attempts: {e}"
                        )
                        return False

                    time.sleep(
                        min(1.0, JSON_RETRY_DELAY * (attempt + 1))
                    )

        except Exception as e:
            print(f"\nWARNING: could not save {path}: {e}")
            return False

        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass


def load_json(path, default):
    path = Path(path)

    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"WARNING: could not read {path}: {e}")
        return default


def cleanup_stale_json_temp_files():
    if not ROOT.exists():
        return

    for pattern in (
        "*.json.tmp",
        "subset.json.*.tmp",
        "failed.json.*.tmp",
        "dead_sources.json.*.tmp",
    ):
        for path in ROOT.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


# ============================================================
# VATEX METADATA
# ============================================================


def download_metadata(url, path):
    if path.exists():
        return

    print(f"Downloading metadata: {url}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with open(path, "wb") as f:
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=path.name,
        ) as bar:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def parse_item(item):
    """
    Example:
        Ptf_2VRj-V0_000122_000132

    YouTube ID: Ptf_2VRj-V0
    start: 122
    end:   132
    """

    clip_id = item["videoID"]
    parts = clip_id.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected VATEX videoID: {clip_id}")

    start = int(parts[-2])
    end = int(parts[-1])
    video_id = "_".join(parts[:-2])

    return {
        "clip_id": clip_id,
        "videoID": video_id,
        "start": start,
        "end": end,
        "url": f"https://www.youtube.com/watch?v={video_id}",
    }


def deterministic_order(items, seed):
    items = list(items)
    random.Random(seed).shuffle(items)
    return items


# ============================================================
# VIDEO VALIDATION
# ============================================================


def validate_video(path):
    path = Path(path)

    if not path.exists():
        return False

    try:
        if path.stat().st_size < 10_000:
            return False
    except OSError:
        return False

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height:format=duration",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )

        data = json.loads(result.stdout)
        streams = data.get("streams", [])

        if not streams:
            return False

        width = streams[0].get("width", 0)
        height = streams[0].get("height", 0)

        if width <= 0 or height <= 0:
            return False

        duration = float(data.get("format", {}).get("duration", 0))
        return duration > 0

    except Exception:
        return False


# ============================================================
# DEPENDENCY / ENVIRONMENT CHECKS
# ============================================================


def check_dependencies():
    for executable in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run(
                [executable, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=30,
            )
        except Exception:
            print(f"ERROR: {executable} is not available in PATH.")
            return False

    try:
        subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=30,
        )
    except Exception:
        print("ERROR: yt-dlp is not installed.")
        print('Run: uv pip install -U "yt-dlp[default]"')
        return False

    try:
        result = subprocess.run(
            [JS_RUNTIME, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
    except Exception:
        print(f"ERROR: {JS_RUNTIME} is not available in this terminal PATH.")
        return False

    return True


def detect_nvenc():
    try:
        encoders = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )

        if "h264_nvenc" not in encoders.stdout:
            return False

        test = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "lavfi",
                "-i", "color=s=64x64:d=0.1",
                "-frames:v", "1",
                "-c:v", "h264_nvenc",
                "-preset", NVENC_PRESET,
                "-f", "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        return test.returncode == 0

    except Exception:
        return False


def encode_args():
    if VIDEO_ENCODER == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc",
            "-preset", NVENC_PRESET,
            "-rc", "vbr",
            "-cq", str(NVENC_CQ),
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
        ]

    return [
        "-c:v", "libx264",
        "-preset", X264_PRESET,
        "-crf", str(X264_CRF),
        "-pix_fmt", "yuv420p",
    ]


# ============================================================
# ERROR CLASSIFICATION
# ============================================================


def shorten_error(message, limit=5000):
    message = (message or "Unknown error").strip()
    return message if len(message) <= limit else message[-limit:]


def permanent_source_error(message):
    """Only strong, video-specific permanent failures go here."""

    text = (message or "").lower()

    markers = (
        "video unavailable",
        "this video is unavailable",
        "private video",
        "this video is private",
        "has been removed by the uploader",
        "this video has been removed",
        "account associated with this video has been terminated",
        "this video is no longer available",
        "members-only content",
    )

    return any(marker in text for marker in markers)


def rate_limit_or_bot_error(message):
    text = (message or "").lower()

    markers = (
        "http error 429",
        "too many requests",
        "sign in to confirm you’re not a bot",
        "sign in to confirm you're not a bot",
        "confirm you’re not a bot",
        "confirm you're not a bot",
    )

    return any(marker in text for marker in markers)


# ============================================================
# SOURCE CACHE
# ============================================================


def clear_partial_source_files(video_id):
    for path in CACHE_DIR.glob(f"{video_id}.*"):
        if path.suffix.lower() in {".part", ".ytdl", ".tmp"}:
            try:
                path.unlink()
            except OSError:
                pass


def cached_source(video_id):
    candidates = []

    for path in CACHE_DIR.glob(f"{video_id}.*"):
        if not path.is_file():
            continue

        if path.suffix.lower() in {".part", ".ytdl", ".tmp"}:
            continue

        candidates.append(path)

    candidates.sort(
        key=lambda p: p.stat().st_size if p.exists() else 0,
        reverse=True,
    )

    for path in candidates:
        if validate_video(path):
            try:
                path.touch()
            except OSError:
                pass
            return path

        try:
            path.unlink()
        except OSError:
            pass

    return None


def prune_cache():
    limit_bytes = int(SOURCE_CACHE_MAX_GB * 1024**3)
    files = []
    total = 0

    try:
        paths = list(CACHE_DIR.iterdir())
    except OSError:
        return

    for path in paths:
        if not path.is_file():
            continue

        if path.suffix.lower() in {".part", ".ytdl", ".tmp"}:
            continue

        try:
            stat = path.stat()
        except OSError:
            continue

        files.append((stat.st_mtime, stat.st_size, path))
        total += stat.st_size

    if total <= limit_bytes:
        return

    files.sort(key=lambda x: x[0])

    for _, size, path in files:
        if total <= limit_bytes:
            break

        if source_is_active(path):
            continue

        try:
            path.unlink()
            total -= size
        except OSError:
            pass


# ============================================================
# SOURCE DOWNLOAD
# ============================================================


def get_source(item):
    """
    Download/reuse one YouTube source.

    Returns:
        (source_path, error_message, permanent_failure)

    If source_path is returned, it has been marked active and must
    be released by process_clip().
    """

    video_id = item["videoID"]

    with get_source_lock(video_id):
        existing = cached_source(video_id)

        if existing is not None:
            acquire_source_use(existing)
            return existing, None, False

        clear_partial_source_files(video_id)

        # Respect a shared cooldown after 429 / bot challenges.
        wait_for_rate_limit_backoff()

        output_template = str(CACHE_DIR / f"{video_id}.%(ext)s")

        # Prefer a video-only stream <= 360p. If YouTube doesn't expose
        # a height-tagged stream, the fallback still requests video-only.
        format_selector = (
            f"bv[height<=?{MAX_HEIGHT}]/"
            f"b[height<=?{MAX_HEIGHT}]/"
            "bv/b"
        )

        command = [
            sys.executable,
            "-m", "yt_dlp",
            "--ignore-config",
            "--no-playlist",
            "--no-progress",
            "--js-runtimes", JS_RUNTIME,
            "--cookies-from-browser", COOKIES_FROM_BROWSER,
            "--retries", str(YTDLP_RETRIES),
            "--fragment-retries", str(YTDLP_FRAGMENT_RETRIES),
            "--extractor-retries", str(YTDLP_EXTRACTOR_RETRIES),
            "--socket-timeout", str(SOCKET_TIMEOUT),
            "--throttled-rate", THROTTLED_RATE,
            "-f", format_selector,
            "-o", output_template,
            item["url"],
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=YTDLP_PROCESS_TIMEOUT,
            )

        except subprocess.TimeoutExpired:
            clear_partial_source_files(video_id)
            return (
                None,
                f"yt-dlp exceeded {YTDLP_PROCESS_TIMEOUT}s hard timeout",
                False,
            )

        except Exception as e:
            clear_partial_source_files(video_id)
            return None, f"yt-dlp process error: {e}", False

        if result.returncode != 0:
            error = shorten_error(
                result.stderr or result.stdout or "yt-dlp failed"
            )

            if rate_limit_or_bot_error(error):
                trigger_rate_limit_backoff()

            permanent = permanent_source_error(error)
            clear_partial_source_files(video_id)
            return None, error, permanent

        source = cached_source(video_id)

        if source is None:
            return (
                None,
                "yt-dlp finished but no valid cached source file was found",
                False,
            )

        acquire_source_use(source)
        return source, None, False


# ============================================================
# CLIP PROCESSING
# ============================================================


def process_clip(item):
    clip_id = item["clip_id"]
    final_path = VIDEO_DIR / f"{clip_id}.mp4"

    if SKIP_EXISTING and final_path.exists():
        if validate_video(final_path):
            return {
                "success": True,
                "existing": True,
                "permanent": False,
            }

        try:
            final_path.unlink()
        except OSError:
            pass

    source_path = None

    try:
        source_path, error, permanent = get_source(item)

        if source_path is None:
            return {
                "success": False,
                "error": shorten_error(error),
                "permanent": permanent,
            }

        duration = item["end"] - item["start"]

        if duration <= 0:
            return {
                "success": False,
                "error": "Invalid VATEX clip duration",
                "permanent": False,
            }

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-ss", str(item["start"]),
            "-i", str(source_path),
            "-t", str(duration),
            "-map", "0:v:0",
            "-an",
            *encode_args(),
            "-movflags", "+faststart",
            str(final_path),
        ]

        with ENCODE_SEMAPHORE:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=180,
            )

        if result.returncode != 0:
            raise RuntimeError(
                shorten_error(result.stderr or "FFmpeg failed")
            )

        if not validate_video(final_path):
            raise RuntimeError("Final MP4 failed ffprobe validation")

        return {
            "success": True,
            "existing": False,
            "permanent": False,
        }

    except subprocess.TimeoutExpired:
        if final_path.exists():
            try:
                final_path.unlink()
            except OSError:
                pass

        return {
            "success": False,
            "error": "FFmpeg exceeded 180s timeout",
            "permanent": False,
        }

    except Exception as e:
        if final_path.exists():
            try:
                final_path.unlink()
            except OSError:
                pass

        return {
            "success": False,
            "error": shorten_error(str(e)),
            "permanent": False,
        }

    finally:
        if source_path is not None:
            release_source_use(source_path)


# ============================================================
# SUBSET / RECOVERY
# ============================================================


def save_subset(train_ids, val_ids):
    return save_json(
        {
            "seed": SEED,
            "train_target": TRAIN_SIZE,
            "validation_target": VAL_SIZE,
            "train_size": len(train_ids),
            "validation_size": len(val_ids),
            "train": train_ids,
            "validation": val_ids,
        },
        SUBSET_FILE,
    )


def load_existing_subset():
    data = load_json(SUBSET_FILE, {})

    if not data or data.get("seed") != SEED:
        return [], []

    return (
        data.get("train", []),
        data.get("validation", []),
    )


def keep_valid_ids(ids):
    valid = []

    for clip_id in ids:
        path = VIDEO_DIR / f"{clip_id}.mp4"
        if validate_video(path):
            valid.append(clip_id)

    return valid


def recover_all_existing_videos(train_items, val_items, train_ids, val_ids):
    """
    Recover valid MP4s already on disk even if an old run crashed
    before writing them into subset.json.
    """

    train_map = {x["clip_id"]: x for x in train_items}
    val_map = {x["clip_id"]: x for x in val_items}

    train_set = set(train_ids)
    val_set = set(val_ids)

    for path in VIDEO_DIR.glob("*.mp4"):
        clip_id = path.stem

        if clip_id in train_map and clip_id not in train_set:
            if validate_video(path):
                train_ids.append(clip_id)
                train_set.add(clip_id)

        elif clip_id in val_map and clip_id not in val_set:
            if validate_video(path):
                val_ids.append(clip_id)
                val_set.add(clip_id)

    return train_ids, val_ids


def trusted_dead_video_ids(dead_sources):
    """
    Return only source IDs that were confirmed permanent under the
    working Firefox + Deno authenticated environment.
    """

    trusted = set()

    for video_id, info in dead_sources.items():
        if not isinstance(info, dict):
            continue

        if TRUST_ONLY_AUTH_VERIFIED_DEAD:
            if info.get("verified_with_auth") is True:
                trusted.add(video_id)
        else:
            trusted.add(video_id)

    return trusted


# ============================================================
# PARALLEL DIRECT DOWNLOADER
# ============================================================


def download_split(
    name,
    candidates,
    target,
    success_ids,
    failed,
    dead_sources,
    train_ids,
    val_ids,
):
    """
    No pre-scan.

    Keep up to WORKERS actual downloads in flight. Each completed
    result is committed immediately. Keep trying candidates until
    exactly target valid clips have been obtained or the split is
    exhausted.
    """

    success_ids = list(success_ids[:target])
    success_set = set(success_ids)
    trusted_dead = trusted_dead_video_ids(dead_sources)

    candidate_iter = iter(candidates)
    exhausted = False
    attempts_this_run = 0

    def next_candidate():
        nonlocal exhausted

        while True:
            try:
                item = next(candidate_iter)
            except StopIteration:
                exhausted = True
                return None

            clip_id = item["clip_id"]
            video_id = item["videoID"]

            if clip_id in success_set:
                continue

            if video_id in trusted_dead:
                continue

            if not RETRY_FAILED and clip_id in failed:
                continue

            return item

    progress = tqdm(
        total=target,
        initial=len(success_ids),
        desc=f"{name} successful",
        unit="clip",
    )

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {}

        def fill_workers():
            nonlocal attempts_this_run

            remaining = target - len(success_ids)
            allowed = min(WORKERS, remaining)

            while len(futures) < allowed and not exhausted:
                item = next_candidate()

                if item is None:
                    break

                future = pool.submit(process_clip, item)
                futures[future] = item
                attempts_this_run += 1

        fill_workers()

        while futures and len(success_ids) < target:
            done, _ = wait(
                futures,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                item = futures.pop(future)
                clip_id = item["clip_id"]
                video_id = item["videoID"]

                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "success": False,
                        "error": f"Worker exception: {e}",
                        "permanent": False,
                    }

                if result.get("success"):
                    if clip_id not in success_set and len(success_ids) < target:
                        success_ids.append(clip_id)
                        success_set.add(clip_id)
                        failed.pop(clip_id, None)

                        # If this source had been marked dead by the old
                        # broken environment, a successful authenticated
                        # download proves that record was a false positive.
                        if video_id in dead_sources and not dead_sources.get(video_id, {}).get("verified_with_auth", False):
                            dead_sources.pop(video_id, None)
                            save_json(dead_sources, DEAD_FILE)

                        progress.update(1)

                        if name == "train":
                            train_ids[:] = success_ids
                        else:
                            val_ids[:] = success_ids

                        save_subset(train_ids, val_ids)
                        save_json(failed, FAILED_FILE)

                else:
                    error = shorten_error(
                        result.get("error", "Unknown error")
                    )
                    permanent = bool(result.get("permanent", False))

                    failed[clip_id] = {
                        "videoID": video_id,
                        "clip_id": clip_id,
                        "start": item["start"],
                        "end": item["end"],
                        "url": item["url"],
                        "error": error,
                        "permanent_source_failure": permanent,
                        "tested_with_auth": True,
                        "js_runtime": JS_RUNTIME,
                        "cookies_from_browser": COOKIES_FROM_BROWSER,
                    }

                    save_json(failed, FAILED_FILE)

                    if permanent:
                        dead_sources[video_id] = {
                            "url": item["url"],
                            "error": error,
                            "verified_with_auth": True,
                            "js_runtime": JS_RUNTIME,
                            "cookies_from_browser": COOKIES_FROM_BROWSER,
                        }

                        trusted_dead.add(video_id)
                        save_json(dead_sources, DEAD_FILE)

                prune_cache()

            fill_workers()

    progress.close()

    print(
        f"{name}: {attempts_this_run} attempted this run; "
        f"{len(success_ids)}/{target} valid"
    )

    if len(success_ids) < target:
        print(
            f"WARNING: candidate list exhausted before reaching "
            f"{target} successful {name} clips."
        )

    return success_ids


# ============================================================
# DESCRIPTION
# ============================================================


def save_description(train_ids, val_ids):
    text = f"""# VATEX Subset

Seed: {SEED}

Train target: {TRAIN_SIZE}
Train successful: {len(train_ids)}

Validation target: {VAL_SIZE}
Validation successful: {len(val_ids)}

Download strategy:
Direct download only (no availability pre-scan)

Workers: {WORKERS}
Encode concurrency: {ENCODE_CONCURRENCY}

YouTube JS runtime: {JS_RUNTIME}
Browser cookies: {COOKIES_FROM_BROWSER}

Source resolution: <= {MAX_HEIGHT}p
Audio: disabled

Encoder: {VIDEO_ENCODER}

yt-dlp process timeout: {YTDLP_PROCESS_TIMEOUT}s
Rate-limit backoff: {RATE_LIMIT_BACKOFF_SECONDS}s

Source cache limit: {SOURCE_CACHE_MAX_GB} GB

subset.json contains only successfully downloaded and
ffprobe-validated clips.

Old dead_sources.json entries are not trusted unless they have
verified_with_auth=true. This allows sources classified while the
old yt-dlp environment was broken to be tried again.
"""

    DESCRIPTION_FILE.write_text(text, encoding="utf-8")


# ============================================================
# MAIN
# ============================================================


def main():
    global VIDEO_ENCODER

    cleanup_stale_json_temp_files()

    if not check_dependencies():
        return

    VIDEO_ENCODER = "h264_nvenc" if detect_nvenc() else "libx264"

    print()
    print("=" * 65)
    print("VATEX DIRECT DOWNLOADER")
    print("=" * 65)
    print(f"Encoder:                 {VIDEO_ENCODER}")
    print(f"Workers:                 {WORKERS}")
    print(f"Encode concurrency:      {ENCODE_CONCURRENCY}")
    print(f"Source resolution:       <= {MAX_HEIGHT}p")
    print(f"JavaScript runtime:      {JS_RUNTIME}")
    print(f"Browser cookies:         {COOKIES_FROM_BROWSER}")
    print(f"yt-dlp hard timeout:     {YTDLP_PROCESS_TIMEOUT}s")
    print(f"Rate-limit backoff:      {RATE_LIMIT_BACKOFF_SECONDS}s")
    print(f"Source cache limit:      {SOURCE_CACHE_MAX_GB} GB")
    print("Availability pre-scan:   disabled")
    print()

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    download_metadata(TRAIN_URL, TRAIN_JSON)
    download_metadata(VAL_URL, VAL_JSON)

    train_raw = load_json(TRAIN_JSON, [])
    val_raw = load_json(VAL_JSON, [])

    train_items = [parse_item(x) for x in train_raw]
    val_items = [parse_item(x) for x in val_raw]

    train_candidates = deterministic_order(train_items, SEED)
    val_candidates = deterministic_order(val_items, SEED + 1)

    print(f"Train annotations:       {len(train_items)}")
    print(f"Validation annotations:  {len(val_items)}")

    # --------------------------------------------------------
    # Recover existing work
    # --------------------------------------------------------

    train_ids, val_ids = load_existing_subset()

    train_ids = keep_valid_ids(train_ids)
    val_ids = keep_valid_ids(val_ids)

    train_ids, val_ids = recover_all_existing_videos(
        train_items,
        val_items,
        train_ids,
        val_ids,
    )

    # Preserve deterministic candidate order for recovered files.
    train_existing = set(train_ids)
    val_existing = set(val_ids)

    train_ids = [
        x["clip_id"]
        for x in train_candidates
        if x["clip_id"] in train_existing
    ][:TRAIN_SIZE]

    val_ids = [
        x["clip_id"]
        for x in val_candidates
        if x["clip_id"] in val_existing
    ][:VAL_SIZE]

    failed = load_json(FAILED_FILE, {})
    dead_sources = load_json(DEAD_FILE, {})

    trusted_dead_count = len(trusted_dead_video_ids(dead_sources))
    untrusted_old_dead_count = max(0, len(dead_sources) - trusted_dead_count)

    print()
    print(f"Recovered valid train clips:       {len(train_ids)}/{TRAIN_SIZE}")
    print(f"Recovered valid validation clips:  {len(val_ids)}/{VAL_SIZE}")
    print(f"Old failed clip records:           {len(failed)}")
    print(f"Trusted auth-verified dead sources:{trusted_dead_count}")
    print(f"Old untrusted dead sources:        {untrusted_old_dead_count}")
    print()

    save_subset(train_ids, val_ids)

    # --------------------------------------------------------
    # Direct train download
    # --------------------------------------------------------

    train_ids = download_split(
        name="train",
        candidates=train_candidates,
        target=TRAIN_SIZE,
        success_ids=train_ids,
        failed=failed,
        dead_sources=dead_sources,
        train_ids=train_ids,
        val_ids=val_ids,
    )

    # --------------------------------------------------------
    # Direct validation download
    # --------------------------------------------------------

    val_ids = download_split(
        name="validation",
        candidates=val_candidates,
        target=VAL_SIZE,
        success_ids=val_ids,
        failed=failed,
        dead_sources=dead_sources,
        train_ids=train_ids,
        val_ids=val_ids,
    )

    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------

    save_subset(train_ids, val_ids)
    save_json(failed, FAILED_FILE)
    save_json(dead_sources, DEAD_FILE)
    save_description(train_ids, val_ids)

    print()
    print("=" * 65)
    print("FINISHED")
    print("=" * 65)
    print(f"Train:            {len(train_ids)}/{TRAIN_SIZE}")
    print(f"Validation:       {len(val_ids)}/{VAL_SIZE}")
    print(f"Failed clips:     {len(failed)}")
    print(
        f"Verified dead:    "
        f"{len(trusted_dead_video_ids(dead_sources))}"
    )
    print(f"Encoder:          {VIDEO_ENCODER}")

    if CLEAR_SOURCE_CACHE_AT_END:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print("Source cache cleared.")


if __name__ == "__main__":
    main()
