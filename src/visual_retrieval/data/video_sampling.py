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
    result = read_video_segments(path, [(start, end)], count_frames)[0]
    if isinstance(result, Exception):
        raise result
    return result


def read_video_segments(path, intervals, count_frames=4):
    """Sample many intervals with two shared passes; return errors per interval.

    Uses the same actual-frame timeline and center selection as single-item
    sampling. Only requested RGB frames are retrieved and retained.
    """
    if not intervals:
        return []
    needs_timestamps = any(a is not None or b is not None for a, b in intervals)
    # Count actually decodable frames with a first pass. Do not trust container metadata.
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    try:
        timestamps = []
        count = 0
        while cap.grab():
            if needs_timestamps:
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            count += 1
    finally:
        cap.release()
    selections = []
    for start, end in intervals:
        try:
            if start is None and end is None:
                indices = middle_indices(count, count_frames)
            elif start is not None and end is not None:
                indices = segment_indices(timestamps, start, end, count_frames)
            else:
                raise ValueError('Both start and end are required for a segment.')
            selections.append(indices)
        except ValueError as error:
            selections.append(error)
    wanted = {index for indices in selections if not isinstance(indices, Exception) for index in indices}
    if not wanted:
        return selections
    cap = cv2.VideoCapture(str(path))
    frames = {}
    frame_errors = {}
    stopped = None
    try:
        for index in range(max(wanted) + 1):
            if not cap.grab():
                stopped = ValueError(f'Video failed while sampling frame {index}: {path}')
                break
            if index in wanted:
                ok, bgr = cap.retrieve()
                if not ok or bgr is None:
                    frame_errors[index] = ValueError(f'Cannot decode frame {index}: {path}')
                else:
                    frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    results = []
    for indices in selections:
        if isinstance(indices, Exception):
            results.append(indices)
            continue
        missing = next((index for index in indices if index not in frames), None)
        if missing is not None:
            results.append(frame_errors.get(missing) or stopped or ValueError(f'Cannot decode frame {missing}: {path}'))
        else:
            results.append((np.stack([frames[index] for index in indices]), indices, count))
    return results
