"""Shared RGB video sampling using actual decoded presentation timestamps."""
import math
import cv2
import numpy as np
from visual_retrieval.common import middle_indices


def segment_indices(timestamps, start, end, count=4):
    """Frame centers among actual presentation times in [start,end)."""
    if (not timestamps or count <= 0 or not math.isfinite(start) or not math.isfinite(end)
            or start < 0 or end <= start):
        raise ValueError('Invalid temporal interval or empty frame timeline.')
    if (any(not math.isfinite(t) or t < 0 for t in timestamps)
            or any(b <= a for a, b in zip(timestamps, timestamps[1:]))):
        raise ValueError('Decoder did not provide increasing frame timestamps; transcode this video to a supported MP4.')
    # A tiny tolerance avoids excluding exact-boundary frames due to floating-point conversion.
    eligible = [i for i, t in enumerate(timestamps) if t >= start - 1e-8 and t < end - 1e-8]
    if not eligible:
        raise ValueError(f'Interval [{start}, {end}) has no decoded frames.')
    return [eligible[index] for index in middle_indices(len(eligible), count)]


def read_video_frames(path, start=None, end=None, count_frames=4):
    # Count actually decodable frames with a first pass. Do not trust container metadata.
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    try:
        timestamps = []
        count = 0
        while cap.grab():
            if start is not None or end is not None:
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            count += 1
    finally:
        cap.release()
    if start is None and end is None:
        indices = middle_indices(count, count_frames)
    elif start is not None and end is not None:
        indices = segment_indices(timestamps, start, end, count_frames)
    else:
        raise ValueError("Both start and end are required for a segment.")
    wanted = set(indices)
    cap = cv2.VideoCapture(str(path))
    frames = {}
    try:
        for index in range(indices[-1] + 1):
            if not cap.grab():
                raise ValueError(f'Video failed while sampling frame {index}: {path}')
            if index in wanted:
                ok, bgr = cap.retrieve()
                if not ok or bgr is None:
                    raise ValueError(f'Cannot decode frame {index}: {path}')
                frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return np.stack([frames[index] for index in indices]), indices, count
