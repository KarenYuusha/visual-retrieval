#!/usr/bin/env python

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_NUM_FRAMES = 12


# ============================================================
# General helpers
# ============================================================

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_existing(root: Path, candidates: List[str]) -> Path:
    for rel in candidates:
        p = root / rel

        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find any of:\n"
        + "\n".join(str(root / x) for x in candidates)
    )


def selected_ids_from_subset(
    subset_path: Path,
    requested_split: Optional[str],
) -> Dict[str, List[str]]:

    subset = load_json(subset_path)

    result = {}

    for key, value in subset.items():
        if (
            isinstance(value, list)
            and all(isinstance(x, str) for x in value)
        ):
            result[key] = value

    if not result:
        raise ValueError(
            f"No split lists found in {subset_path}"
        )

    if requested_split is None:
        return result

    aliases = {
        "train": {"train"},
        "test": {"test"},
        "val": {"val", "validation"},
        "validation": {"val", "validation"},
    }

    wanted = aliases.get(
        requested_split,
        {requested_split},
    )

    filtered = {
        key: value
        for key, value in result.items()
        if key.lower() in wanted
    }

    if not filtered:
        raise ValueError(
            f"Split '{requested_split}' not found. "
            f"Available: {list(result.keys())}"
        )

    return filtered


def video_id_variants(video_id: str):

    variants = [video_id]

    if video_id.startswith("v_"):
        variants.append(video_id[2:])
    else:
        variants.append("v_" + video_id)

    return list(dict.fromkeys(variants))


def resolve_video_path(
    raw_videos: Path,
    video_id: str,
):

    extensions = [
        ".mp4",
        ".mkv",
        ".webm",
        ".avi",
        ".mov",
    ]

    for stem in video_id_variants(video_id):

        for ext in extensions:

            path = raw_videos / f"{stem}{ext}"

            if path.exists():
                return path

    return None


def safe_h5_key(item_id: str):
    return item_id.replace("/", "__")


# ============================================================
# MSR-VTT
# ============================================================

def iter_msrvtt(
    root: Path,
    requested_split: Optional[str],
):

    subset_path = root / "subset.json"
    raw_videos = root / "raw_videos"

    ann_path = find_existing(
        root,
        [
            "raw_data/MSRVTT_data.json",
            "raw_data/msrvtt_data.json",
            "MSRVTT_data.json",
        ],
    )

    split_ids = selected_ids_from_subset(
        subset_path,
        requested_split,
    )

    annotation = load_json(ann_path)

    caption_map = defaultdict(list)

    for row in annotation.get("sentences", []):

        video_id = row.get("video_id")
        caption = row.get("caption")

        if video_id and caption:
            caption_map[str(video_id)].append(
                str(caption)
            )

    for split, ids in split_ids.items():

        for video_id in ids:

            video_path = resolve_video_path(
                raw_videos,
                video_id,
            )

            if video_path is None:

                print(
                    f"[WARN] Missing MSR-VTT video: "
                    f"{video_id}"
                )

                continue

            yield {
                "item_id": video_id,
                "video_id": video_id,
                "video_path": video_path,
                "split": split,
                "start": None,
                "end": None,
                "captions":
                    caption_map.get(video_id, []),
            }


# ============================================================
# VATEX
# ============================================================

def iter_vatex(
    root: Path,
    requested_split: Optional[str],
):

    subset_path = root / "subset.json"
    raw_videos = root / "raw_videos"

    split_ids = selected_ids_from_subset(
        subset_path,
        requested_split,
    )

    annotation_candidates = [
        "raw_data/vatex_training_v1.0.json",
        "raw_data/vatex_validation_v1.0.json",
        "vatex_training_v1.0.json",
        "vatex_validation_v1.0.json",
    ]

    annotation_map = {}

    for rel in annotation_candidates:

        path = root / rel

        if not path.exists():
            continue

        data = load_json(path)

        for row in data:

            video_id = row.get("videoID")

            if video_id:
                annotation_map[str(video_id)] = row

    if not annotation_map:

        raise FileNotFoundError(
            "VATEX annotation files were not found."
        )

    for split, ids in split_ids.items():

        for video_id in ids:

            video_path = resolve_video_path(
                raw_videos,
                video_id,
            )

            if video_path is None:

                print(
                    f"[WARN] Missing VATEX video: "
                    f"{video_id}"
                )

                continue

            row = annotation_map.get(
                video_id,
                {},
            )

            captions = row.get(
                "enCap",
                [],
            )

            if not isinstance(captions, list):
                captions = []

            yield {
                "item_id": video_id,
                "video_id": video_id,
                "video_path": video_path,
                "split": split,
                "start": None,
                "end": None,
                "captions": captions,
            }


# ============================================================
# ActivityNet Captions
# ============================================================

def activitynet_lookup(
    annotation,
    video_id,
):

    for candidate in video_id_variants(video_id):

        if candidate in annotation:
            return candidate, annotation[candidate]

    return None, None


def iter_activitynet(
    root: Path,
    requested_split: Optional[str],
    val_file: str,
):

    subset_path = root / "subset.json"
    raw_videos = root / "raw_videos"

    split_ids = selected_ids_from_subset(
        subset_path,
        requested_split,
    )

    train_path = root / "raw_data" / "train.json"

    if not train_path.exists():
        train_path = root / "train.json"

    val_path = root / "raw_data" / val_file

    if not val_path.exists():
        val_path = root / val_file

    train_ann = (
        load_json(train_path)
        if train_path.exists()
        else {}
    )

    val_ann = (
        load_json(val_path)
        if val_path.exists()
        else {}
    )

    for split, ids in split_ids.items():

        split_lower = split.lower()

        if split_lower == "train":
            annotation = train_ann

        elif split_lower in {
            "val",
            "validation",
            "test",
        }:
            annotation = val_ann

        else:
            continue

        for subset_video_id in ids:

            canonical_id, row = activitynet_lookup(
                annotation,
                subset_video_id,
            )

            if row is None:

                print(
                    f"[WARN] Annotation missing: "
                    f"{subset_video_id}"
                )

                continue

            video_path = resolve_video_path(
                raw_videos,
                canonical_id,
            )

            if video_path is None:

                print(
                    f"[WARN] Video missing: "
                    f"{canonical_id}"
                )

                continue

            timestamps = row.get(
                "timestamps",
                [],
            )

            sentences = row.get(
                "sentences",
                [],
            )

            for segment_idx, (
                timestamp,
                sentence,
            ) in enumerate(
                zip(timestamps, sentences)
            ):

                if (
                    not isinstance(
                        timestamp,
                        (list, tuple),
                    )
                    or len(timestamp) != 2
                ):
                    continue

                start = float(timestamp[0])
                end = float(timestamp[1])

                if (
                    not math.isfinite(start)
                    or not math.isfinite(end)
                    or end <= start
                ):
                    continue

                yield {
                    "item_id":
                        f"{canonical_id}"
                        f"__seg{segment_idx:04d}",

                    "video_id":
                        canonical_id,

                    "video_path":
                        video_path,

                    "split":
                        split,

                    "start":
                        start,

                    "end":
                        end,

                    "captions":
                        [str(sentence)],

                    "segment_index":
                        segment_idx,
                }


# ============================================================
# Video information
# ============================================================

def get_video_info(cap):

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    frame_count = float(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:

        raise RuntimeError(
            "Invalid FPS"
        )

    if frame_count <= 0:

        raise RuntimeError(
            "Invalid frame count"
        )

    duration = frame_count / fps

    return fps, duration


# ============================================================
# OpenCV decoder
# ============================================================

def read_frame_opencv(
    cap,
    time_sec,
    fps,
):

    # First try timestamp seeking
    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        time_sec * 1000.0,
    )

    success, frame = cap.read()

    # Fallback to frame-number seeking
    if not success:

        frame_index = int(
            round(time_sec * fps)
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        success, frame = cap.read()

    if not success or frame is None:
        return None

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(frame)


# ============================================================
# FFmpeg fallback decoder
# ============================================================

def read_frame_ffmpeg(
    video_path: Path,
    time_sec: float,
):

    """
    Decode a single frame at a timestamp.

    -ss comes AFTER -i for accurate seeking.
    This is slower, but it is only used when
    OpenCV has already failed.
    """

    command = [
        "ffmpeg",
        "-loglevel",
        "error",

        "-i",
        str(video_path),

        "-ss",
        f"{time_sec:.6f}",

        "-frames:v",
        "1",

        "-f",
        "image2pipe",

        "-vcodec",
        "png",

        "-"
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )

    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        return None

    if not result.stdout:
        return None

    try:

        image = Image.open(
            io.BytesIO(
                result.stdout
            )
        )

        return image.convert("RGB")

    except Exception:
        return None


def read_frame_ffmpeg_robust(
    video_path,
    time_sec,
):

    """
    Try the requested timestamp first.

    If that fails, try timestamps extremely
    close to it.
    """

    offsets = [
        0.0,
        -0.10,
        +0.10,
        -0.25,
        +0.25,
        -0.50,
        +0.50,
    ]

    for offset in offsets:

        candidate_time = max(
            0.0,
            time_sec + offset,
        )

        frame = read_frame_ffmpeg(
            video_path,
            candidate_time,
        )

        if frame is not None:

            return (
                frame,
                candidate_time,
            )

    return None, None


# ============================================================
# Uniform sampling
# ============================================================

def sample_uniform_frames(
    video_path: Path,
    num_frames: int,
    start=None,
    end=None,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: "
            f"{video_path}"
        )

    try:

        fps, duration = get_video_info(
            cap
        )

        if start is None:
            actual_start = 0.0
        else:
            actual_start = max(
                0.0,
                float(start),
            )

        if end is None:
            actual_end = duration
        else:
            actual_end = min(
                float(end),
                duration,
            )

        # Avoid asking for a frame exactly at EOF
        safety_margin = (
            0.5
            / max(fps, 1.0)
        )

        actual_end = min(
            actual_end,
            duration - safety_margin,
        )

        if actual_end <= actual_start:

            raise RuntimeError(
                f"Invalid segment: "
                f"{actual_start:.3f}s -> "
                f"{actual_end:.3f}s "
                f"(video duration "
                f"{duration:.3f}s)"
            )

        # Divide segment into equal bins
        edges = np.linspace(
            actual_start,
            actual_end,
            num_frames + 1,
        )

        # Sample center of each bin
        requested_times = (
            edges[:-1]
            + edges[1:]
        ) / 2.0

        frames = []
        used_times = []

        for requested_time in requested_times:

            requested_time = float(
                requested_time
            )

            # ----------------------------------------
            # Try OpenCV
            # ----------------------------------------

            frame = read_frame_opencv(
                cap,
                requested_time,
                fps,
            )

            if frame is not None:

                frames.append(frame)

                used_times.append(
                    requested_time
                )

                continue

            print(
                f"\n[FALLBACK] "
                f"OpenCV failed\n"
                f"  video: "
                f"{video_path.name}\n"
                f"  time:  "
                f"{requested_time:.3f}s"
            )

            # ----------------------------------------
            # Try FFmpeg
            # ----------------------------------------

            frame, used_time = (
                read_frame_ffmpeg_robust(
                    video_path,
                    requested_time,
                )
            )

            if frame is None:

                raise RuntimeError(
                    "OpenCV and FFmpeg "
                    "both failed at "
                    f"{requested_time:.3f}s"
                )

            print(
                f"[FALLBACK OK] "
                f"{requested_time:.3f}s "
                f"-> {used_time:.3f}s"
            )

            frames.append(frame)

            used_times.append(
                used_time
            )

        if len(frames) != num_frames:

            raise RuntimeError(
                f"Expected {num_frames} frames, "
                f"got {len(frames)}"
            )

        return (
            frames,
            np.asarray(
                used_times,
                dtype=np.float32,
            ),
        )

    finally:

        cap.release()


# ============================================================
# CLIP encoder
# ============================================================

class ClipEncoder:

    def __init__(self, device):

        self.device = torch.device(
            device
        )

        self.processor = (
            CLIPProcessor.from_pretrained(
                MODEL_ID
            )
        )

        self.model = (
            CLIPModel.from_pretrained(
                MODEL_ID
            )
            .to(self.device)
        )

        self.model.eval()


    @torch.inference_mode()
    def encode_images(self, images):

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        pixel_values = (
            inputs["pixel_values"]
            .to(self.device)
        )

        if self.device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                outputs = (
                    self.model.vision_model(
                        pixel_values=pixel_values
                    )
                )

                features = (
                    self.model.visual_projection(
                        outputs.pooler_output
                    )
                )

        else:

            outputs = (
                self.model.vision_model(
                    pixel_values=pixel_values
                )
            )

            features = (
                self.model.visual_projection(
                    outputs.pooler_output
                )
            )

        features = features.float()

        return F.normalize(
            features,
            p=2,
            dim=-1,
        )


# ============================================================
# HDF5
# ============================================================

def existing_complete(
    h5,
    item_id,
):

    key = safe_h5_key(
        item_id
    )

    return (
        key in h5
        and "frames" in h5[key]
        and "mean" in h5[key]
        and "sample_times" in h5[key]
    )


def write_item(
    h5,
    item,
    frame_features,
    mean_feature,
    sample_times,
):

    key = safe_h5_key(
        item["item_id"]
    )

    if key in h5:
        del h5[key]

    group = h5.create_group(
        key
    )

    group.create_dataset(
        "frames",
        data=frame_features.astype(
            np.float16
        ),
    )

    group.create_dataset(
        "mean",
        data=mean_feature.astype(
            np.float32
        ),
    )

    group.create_dataset(
        "sample_times",
        data=sample_times.astype(
            np.float32
        ),
    )

    group.attrs["item_id"] = str(
        item["item_id"]
    )

    group.attrs["video_id"] = str(
        item["video_id"]
    )

    group.attrs["split"] = str(
        item["split"]
    )

    group.attrs["video_path"] = str(
        item["video_path"]
    )

    group.attrs["start"] = (
        np.nan
        if item.get("start") is None
        else float(item["start"])
    )

    group.attrs["end"] = (
        np.nan
        if item.get("end") is None
        else float(item["end"])
    )

    group.attrs["captions_json"] = (
        json.dumps(
            item.get(
                "captions",
                [],
            ),
            ensure_ascii=False,
        )
    )

    if "segment_index" in item:

        group.attrs["segment_index"] = int(
            item["segment_index"]
        )

    h5.flush()


# ============================================================
# Main
# ============================================================

def run(args):

    root = Path(
        args.root
    ).resolve()

    feature_dir = (
        root
        / "features"
    )

    feature_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.output:

        output_path = Path(
            args.output
        ).resolve()

    else:

        output_path = (
            feature_dir
            / (
                f"clip_vit_b32_"
                f"{args.num_frames}f_mean.h5"
            )
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    if args.dataset == "msrvtt":

        items = iter_msrvtt(
            root,
            args.split,
        )

    elif args.dataset == "vatex":

        items = iter_vatex(
            root,
            args.split,
        )

    elif args.dataset == "activitynet":

        items = iter_activitynet(
            root,
            args.split,
            args.activitynet_val_file,
        )

    else:

        raise ValueError(
            args.dataset
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = args.device

    if (
        device == "cuda"
        and not torch.cuda.is_available()
    ):

        print(
            "[WARN] CUDA unavailable. "
            "Using CPU."
        )

        device = "cpu"

    print()
    print(f"Dataset:     {args.dataset}")
    print(f"Root:        {root}")
    print(f"Output:      {output_path}")
    print(f"Model:       {MODEL_ID}")
    print(f"Frames:      {args.num_frames}")
    print(f"Device:      {device}")
    print(
        f"Split:       "
        f"{args.split or 'all'}"
    )
    print(
        "Decoder:     "
        "OpenCV -> FFmpeg fallback"
    )
    print()

    encoder = ClipEncoder(
        device
    )

    processed = 0
    skipped = 0
    failed = 0

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    with h5py.File(
        output_path,
        "a",
    ) as h5:

        h5.attrs["model_id"] = MODEL_ID

        h5.attrs["num_frames"] = (
            args.num_frames
        )

        h5.attrs["sampling"] = (
            "uniform_midpoint"
        )

        h5.attrs["pooling"] = (
            "mean"
        )

        h5.attrs["decoder"] = (
            "opencv_then_ffmpeg"
        )

        pbar = tqdm(
            items,
            desc=f"CLIP {args.dataset}",
            unit="item",
        )

        for item in pbar:

            # Existing successful item
            if existing_complete(
                h5,
                item["item_id"],
            ):

                skipped += 1

                continue

            if (
                args.max_items is not None
                and processed >= args.max_items
            ):
                break

            try:

                frames, sample_times = (
                    sample_uniform_frames(
                        video_path=
                            item["video_path"],

                        num_frames=
                            args.num_frames,

                        start=
                            item.get("start"),

                        end=
                            item.get("end"),
                    )
                )

                frame_features = (
                    encoder.encode_images(
                        frames
                    )
                )

                # Average the normalized
                # frame embeddings
                video_feature = (
                    frame_features.mean(
                        dim=0
                    )
                )

                # Normalize final video vector
                video_feature = (
                    F.normalize(
                        video_feature,
                        p=2,
                        dim=0,
                    )
                )

                write_item(
                    h5=h5,

                    item=item,

                    frame_features=(
                        frame_features
                        .cpu()
                        .numpy()
                    ),

                    mean_feature=(
                        video_feature
                        .cpu()
                        .numpy()
                    ),

                    sample_times=
                        sample_times,
                )

                processed += 1

            except Exception as e:

                failed += 1

                pbar.write(
                    "\n[FAIL]"
                    f"\n  item:  "
                    f"{item['item_id']}"
                    f"\n  video: "
                    f"{item['video_path']}"
                    f"\n  error: "
                    f"{e}\n"
                )

            pbar.set_postfix(
                done=processed,
                skip=skipped,
                fail=failed,
            )

    print()
    print("Finished.")
    print(
        f"New items:     {processed}"
    )
    print(
        f"Already done:  {skipped}"
    )
    print(
        f"Failed:        {failed}"
    )
    print(
        f"Saved to:      {output_path}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "msrvtt",
            "vatex",
            "activitynet",
        ],
    )

    parser.add_argument(
        "--root",
        required=True,
    )

    parser.add_argument(
        "--split",
        default=None,
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--device",
        choices=[
            "cuda",
            "cpu",
        ],
        default="cuda",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--activitynet-val-file",
        default="val_1.json",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(
        parse_args()
    )