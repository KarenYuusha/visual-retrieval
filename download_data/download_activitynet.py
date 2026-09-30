import json
import random
import subprocess
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("activitynet_captions")

# Number of SUCCESSFULLY downloaded videos wanted
# TRAIN_SIZE = 5
# VAL_SIZE = 5

# Later:
TRAIN_SIZE = 500
VAL_SIZE = 500

SEED = 67

# Direct HTTPS downloads, so a few parallel workers are fine.
WORKERS = 2

# Number of times to retry an individual HTTP download.
DOWNLOAD_RETRIES = 2

SKIP_EXISTING = True


# ============================================================
# PATHS
# ============================================================

RAW_DATA_DIR = ROOT / "raw_data"
VIDEO_DIR = ROOT / "raw_videos"

SUBSET_FILE = ROOT / "subset.json"
FAILED_FILE = ROOT / "failed.json"
DESCRIPTION_FILE = ROOT / "description.md"

CAPTIONS_ZIP = RAW_DATA_DIR / "captions.zip"

TRAIN_JSON = RAW_DATA_DIR / "train.json"
VAL_JSON = RAW_DATA_DIR / "val_1.json"
TEST_JSON = RAW_DATA_DIR / "val_2.json"

VIDEO_SOURCES_FILE = (
    RAW_DATA_DIR
    / "video_sources.jsonl"
)


RAW_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VIDEO_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SOURCES
# ============================================================

# Official ActivityNet Captions annotation archive
CAPTIONS_URL = (
    "https://cs.stanford.edu/people/"
    "ranjaykrishna/densevid/captions.zip"
)

# Direct MP4 mirror metadata
VIDEO_METADATA_URL = (
    "https://huggingface.co/datasets/"
    "TornadoLabs/activitynet/"
    "resolve/main/metadata.jsonl"
)


# ============================================================
# JSON HELPERS
# ============================================================

def save_json(data, path):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# DOWNLOAD SMALL METADATA FILE
# ============================================================

def download_file(url, path):
    if path.exists():
        print(
            f"Already exists: {path}"
        )
        return

    print(
        f"Downloading: {url}"
    )

    response = requests.get(
        url,
        stream=True,
        timeout=(15, 120),
    )

    response.raise_for_status()

    total = int(
        response.headers.get(
            "content-length",
            0,
        )
    )

    with open(
        path,
        "wb",
    ) as f:

        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=path.name,
        ) as bar:

            for chunk in (
                response.iter_content(
                    chunk_size=1024 * 1024
                )
            ):

                if not chunk:
                    continue

                f.write(chunk)
                bar.update(
                    len(chunk)
                )


# ============================================================
# DOWNLOAD + EXTRACT CAPTIONS
# ============================================================

def prepare_annotations():

    if (
        TRAIN_JSON.exists()
        and VAL_JSON.exists()
        and TEST_JSON.exists()
    ):
        print(
            "ActivityNet Captions "
            "annotations already exist."
        )

        return


    download_file(
        CAPTIONS_URL,
        CAPTIONS_ZIP,
    )


    print(
        "Extracting captions.zip..."
    )


    with zipfile.ZipFile(
        CAPTIONS_ZIP,
        "r",
    ) as archive:

        archive.extractall(
            RAW_DATA_DIR
        )


    if not TRAIN_JSON.exists():
        raise RuntimeError(
            "train.json was not found "
            "after extraction."
        )

    if not VAL_JSON.exists():
        raise RuntimeError(
            "val_1.json was not found "
            "after extraction."
        )


    print(
        "Annotations extracted."
    )


# ============================================================
# LOAD VIDEO MIRROR
# ============================================================

def prepare_video_sources():

    download_file(
        VIDEO_METADATA_URL,
        VIDEO_SOURCES_FILE,
    )


def load_video_sources():

    sources = {}

    with open(
        VIDEO_SOURCES_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(
                line
            )

            video_id = item[
                "video_id"
            ]

            sources[
                video_id
            ] = {
                "video_id":
                    video_id,

                "video_url":
                    item["video_url"],

                "youtube_url":
                    item.get(
                        "youtube_url"
                    ),

                "size_bytes":
                    item.get(
                        "size_bytes"
                    ),
            }

    return sources


# ============================================================
# ACTIVITYNET ID HELPERS
# ============================================================

def annotation_to_youtube_id(
    annotation_id
):
    """
    ActivityNet Captions IDs look like:

        v_QOlSCBRmfWY

    TornadoLabs metadata uses:

        QOlSCBRmfWY
    """

    if annotation_id.startswith(
        "v_"
    ):
        return annotation_id[2:]

    return annotation_id


# ============================================================
# DETERMINISTIC CANDIDATE ORDER
# ============================================================

def build_candidates(
    annotations,
    sources,
    seed,
):

    candidates = []

    for annotation_id in (
        annotations.keys()
    ):

        youtube_id = (
            annotation_to_youtube_id(
                annotation_id
            )
        )

        source = sources.get(
            youtube_id
        )

        # Only use videos actually available
        # in the direct MP4 mirror.
        if source is None:
            continue


        candidates.append(
            {
                "activitynet_id":
                    annotation_id,

                "youtube_id":
                    youtube_id,

                "video_url":
                    source["video_url"],

                "size_bytes":
                    source.get(
                        "size_bytes"
                    ),
            }
        )


    random.Random(
        seed
    ).shuffle(
        candidates
    )


    return candidates


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


    command = [
        "ffprobe",

        "-v",
        "error",

        "-select_streams",
        "v:0",

        "-show_entries",
        (
            "stream=width,height:"
            "format=duration"
        ),

        "-of",
        "json",

        str(path),
    ]


    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )

        data = json.loads(
            result.stdout
        )

        streams = data.get(
            "streams",
            [],
        )


        if not streams:
            return False


        width = streams[0].get(
            "width",
            0,
        )

        height = streams[0].get(
            "height",
            0,
        )


        if width <= 0 or height <= 0:
            return False


        duration = float(
            data.get(
                "format",
                {},
            ).get(
                "duration",
                0,
            )
        )


        return duration > 0


    except Exception:

        return False


# ============================================================
# DOWNLOAD ONE VIDEO
# ============================================================

def download_video(item):

    activitynet_id = item[
        "activitynet_id"
    ]

    url = item[
        "video_url"
    ]

    final_path = (
        VIDEO_DIR
        / f"{activitynet_id}.mp4"
    )

    part_path = (
        VIDEO_DIR
        / f"{activitynet_id}.mp4.part"
    )


    # --------------------------------------------------------
    # REUSE EXISTING VIDEO
    # --------------------------------------------------------

    if (
        SKIP_EXISTING
        and final_path.exists()
    ):

        if validate_video(
            final_path
        ):

            return {
                "success": True,
                "existing": True,
            }


        try:
            final_path.unlink()
        except OSError:
            pass


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    last_error = None


    for attempt in range(
        DOWNLOAD_RETRIES + 1
    ):

        try:

            if part_path.exists():

                try:
                    part_path.unlink()
                except OSError:
                    pass


            with requests.get(
                url,
                stream=True,
                timeout=(15, 180),
            ) as response:

                response.raise_for_status()


                with open(
                    part_path,
                    "wb",
                ) as f:

                    for chunk in (
                        response.iter_content(
                            chunk_size=1024 * 1024
                        )
                    ):

                        if not chunk:
                            continue

                        f.write(chunk)


            # Rename only after complete HTTP
            # download.
            part_path.replace(
                final_path
            )


            # Verify actual MP4.
            if not validate_video(
                final_path
            ):

                raise RuntimeError(
                    "Downloaded file failed "
                    "ffprobe validation."
                )


            return {
                "success": True,
                "existing": False,
            }


        except Exception as e:

            last_error = str(e)


            if part_path.exists():

                try:
                    part_path.unlink()
                except OSError:
                    pass


            if final_path.exists():

                if not validate_video(
                    final_path
                ):

                    try:
                        final_path.unlink()
                    except OSError:
                        pass


    return {
        "success": False,
        "error": last_error,
    }


# ============================================================
# SUBSET STATE
# ============================================================

def save_subset(
    train_ids,
    val_ids,
):

    subset = {
        "seed": SEED,

        "train_target":
            TRAIN_SIZE,

        "validation_target":
            VAL_SIZE,

        "train_size":
            len(train_ids),

        "validation_size":
            len(val_ids),

        "train":
            train_ids,

        "validation":
            val_ids,
    }


    save_json(
        subset,
        SUBSET_FILE,
    )


def load_existing_subset():

    if not SUBSET_FILE.exists():

        return [], []


    data = load_json(
        SUBSET_FILE
    )


    if data.get(
        "seed"
    ) != SEED:

        return [], []


    return (
        data.get(
            "train",
            [],
        ),

        data.get(
            "validation",
            [],
        ),
    )


def keep_valid(ids):

    valid = []


    for activitynet_id in ids:

        path = (
            VIDEO_DIR
            / f"{activitynet_id}.mp4"
        )

        if validate_video(
            path
        ):

            valid.append(
                activitynet_id
            )


    return valid


# ============================================================
# DOWNLOAD UNTIL EXACT SUCCESS COUNT
# ============================================================

def download_split(
    split_name,
    candidates,
    target,
    success_ids,
    failed,
    train_ids,
    val_ids,
):

    success_ids = list(
        success_ids[:target]
    )

    success_set = set(
        success_ids
    )


    candidate_iter = iter(
        candidates
    )

    exhausted = False


    progress = tqdm(
        total=target,
        initial=len(
            success_ids
        ),
        desc=(
            f"{split_name} successful"
        ),
        unit="video",
    )


    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as pool:

        futures = {}


        def fill_workers():

            nonlocal exhausted


            while (
                len(futures) < WORKERS
                and not exhausted
                and (
                    len(success_ids)
                    + len(futures)
                    < target
                )
            ):

                try:

                    item = next(
                        candidate_iter
                    )

                except StopIteration:

                    exhausted = True
                    break


                activitynet_id = item[
                    "activitynet_id"
                ]


                if (
                    activitynet_id
                    in success_set
                ):
                    continue


                future = pool.submit(
                    download_video,
                    item,
                )


                futures[
                    future
                ] = item


        fill_workers()


        while (
            futures
            and len(success_ids) < target
        ):

            done, _ = wait(
                futures,
                return_when=(
                    FIRST_COMPLETED
                ),
            )


            for future in done:

                item = futures.pop(
                    future
                )

                activitynet_id = item[
                    "activitynet_id"
                ]


                try:

                    result = (
                        future.result()
                    )

                except Exception as e:

                    result = {
                        "success": False,
                        "error": str(e),
                    }


                if result.get(
                    "success"
                ):

                    if (
                        activitynet_id
                        not in success_set
                    ):

                        success_ids.append(
                            activitynet_id
                        )

                        success_set.add(
                            activitynet_id
                        )

                        failed.pop(
                            activitynet_id,
                            None,
                        )

                        progress.update(
                            1
                        )


                        if (
                            split_name
                            == "train"
                        ):

                            train_ids[:] = (
                                success_ids
                            )

                        else:

                            val_ids[:] = (
                                success_ids
                            )


                        save_subset(
                            train_ids,
                            val_ids,
                        )


                else:

                    failed[
                        activitynet_id
                    ] = {
                        "youtube_id":
                            item[
                                "youtube_id"
                            ],

                        "video_url":
                            item[
                                "video_url"
                            ],

                        "error":
                            result.get(
                                "error",
                                "Unknown error",
                            ),
                    }


                    save_json(
                        failed,
                        FAILED_FILE,
                    )


            fill_workers()


    progress.close()


    return success_ids


# ============================================================
# DESCRIPTION
# ============================================================

def save_description(
    train_ids,
    val_ids,
):

    text = f"""# ActivityNet Captions Subset

Dataset:
ActivityNet Captions

Annotations:
Official ActivityNet Captions train.json and val_1.json.

Video source:
TornadoLabs/activitynet direct MP4 mirror.

Seed:
{SEED}

Train target:
{TRAIN_SIZE}

Train successfully downloaded:
{len(train_ids)}

Validation target:
{VAL_SIZE}

Validation successfully downloaded:
{len(val_ids)}

## IDs

Files retain the original ActivityNet Captions identifier.

Example:

v_QOlSCBRmfWY.mp4

## Captions

train.json and val_1.json contain:

- duration
- timestamps
- sentences

Each video contains multiple temporally localized captions.

## Evaluation

For a whole-video retrieval baseline, the segment sentences can
be concatenated to form one paragraph query for the video.

subset.json defines the exact videos used in this project.
"""

    DESCRIPTION_FILE.write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# DEPENDENCY CHECK
# ============================================================

def check_ffprobe():

    try:

        subprocess.run(
            [
                "ffprobe",
                "-version",
            ],

            stdout=(
                subprocess.DEVNULL
            ),

            stderr=(
                subprocess.DEVNULL
            ),

            check=True,
            timeout=20,
        )

        return True


    except Exception:

        print(
            "ERROR: ffprobe not found "
            "in PATH."
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    if not check_ffprobe():
        return


    # --------------------------------------------------------
    # PREPARE RAW DATA
    # --------------------------------------------------------

    prepare_annotations()

    prepare_video_sources()


    # --------------------------------------------------------
    # LOAD ANNOTATIONS
    # --------------------------------------------------------

    train_annotations = load_json(
        TRAIN_JSON
    )

    val_annotations = load_json(
        VAL_JSON
    )


    print()
    print(
        f"Official train videos: "
        f"{len(train_annotations)}"
    )

    print(
        f"Official validation videos: "
        f"{len(val_annotations)}"
    )


    # --------------------------------------------------------
    # LOAD DIRECT VIDEO SOURCES
    # --------------------------------------------------------

    sources = load_video_sources()


    print(
        f"Directly hosted videos: "
        f"{len(sources)}"
    )


    # --------------------------------------------------------
    # INTERSECT OFFICIAL IDS WITH AVAILABLE MP4S
    # --------------------------------------------------------

    train_candidates = (
        build_candidates(
            train_annotations,
            sources,
            SEED,
        )
    )

    val_candidates = (
        build_candidates(
            val_annotations,
            sources,
            SEED + 1,
        )
    )


    print()
    print(
        f"Train videos available "
        f"in mirror: "
        f"{len(train_candidates)}"
    )

    print(
        f"Validation videos available "
        f"in mirror: "
        f"{len(val_candidates)}"
    )


    # --------------------------------------------------------
    # RESUME OLD SUBSET
    # --------------------------------------------------------

    train_ids, val_ids = (
        load_existing_subset()
    )


    train_ids = (
        keep_valid(
            train_ids
        )[:TRAIN_SIZE]
    )

    val_ids = (
        keep_valid(
            val_ids
        )[:VAL_SIZE]
    )


    # --------------------------------------------------------
    # FAILURE STATE
    # --------------------------------------------------------

    if FAILED_FILE.exists():

        failed = load_json(
            FAILED_FILE
        )

    else:

        failed = {}


    save_subset(
        train_ids,
        val_ids,
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_ids = download_split(
        split_name="train",

        candidates=train_candidates,

        target=TRAIN_SIZE,

        success_ids=train_ids,

        failed=failed,

        train_ids=train_ids,

        val_ids=val_ids,
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_ids = download_split(
        split_name="validation",

        candidates=val_candidates,

        target=VAL_SIZE,

        success_ids=val_ids,

        failed=failed,

        train_ids=train_ids,

        val_ids=val_ids,
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_subset(
        train_ids,
        val_ids,
    )

    save_json(
        failed,
        FAILED_FILE,
    )

    save_description(
        train_ids,
        val_ids,
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "ACTIVITYNET CAPTIONS DOWNLOAD COMPLETE"
    )
    print("=" * 60)

    print(
        f"Train:      "
        f"{len(train_ids)}/"
        f"{TRAIN_SIZE}"
    )

    print(
        f"Validation: "
        f"{len(val_ids)}/"
        f"{VAL_SIZE}"
    )

    print(
        f"Failed:     "
        f"{len(failed)}"
    )

    print()
    print(
        f"Videos: {VIDEO_DIR}"
    )

    print(
        f"Subset: {SUBSET_FILE}"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()