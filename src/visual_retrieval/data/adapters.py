"""Strict adapters for the downloaded subsets; one combined gallery per dataset."""
import json
import math
from pathlib import Path
import re

from visual_retrieval.common import PROTOCOL, find_annotations, load_captions, load_subset, middle_indices

DATASETS = ('msrvtt', 'vatex', 'activitynet', 'activitynet_captions')
ROOT_NAMES = dict(msrvtt='msr_vtt', vatex='vatex', activitynet='activitynet_captions', activitynet_captions='activitynet_captions')
SPLIT_NAMES = ('train', 'test', 'val', 'validation')


from .manifests import Dataset


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def combined_subset(path):
    data = read_json(path)
    ids, splits = [], []
    for split in SPLIT_NAMES:
        if split not in data:
            continue
        if not isinstance(data[split], list):
            raise ValueError(f'{path}: {split} must be a list of video IDs.')
        splits.append(split)
        for value in data[split]:
            if not isinstance(value, str):
                raise ValueError(f'{path}: expected video ID strings.')
            value = value.removesuffix('.mp4')
            if not value or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
                raise ValueError(f'Unsafe video ID: {value!r}')
            if value not in ids:
                ids.append(value)
    if not ids:
        raise ValueError(f'{path}: no IDs in train/test/val/validation.')
    return ids, splits


def annotation_files(root, names, explicit):
    if explicit:
        paths = [Path(p) for p in explicit]
    else:
        paths = []
        for name in names:
            for prefix in ('raw_data', ''):
                path = root / prefix / name
                if path.is_file():
                    paths.append(path)
                    break
    if not paths:
        raise FileNotFoundError(f'No annotations found under {root}; expected {names}, or use --annotations PATH [PATH ...].')
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def caption_list(values, video_id):
    if not isinstance(values, list) or not values:
        raise ValueError(f'{video_id}: no caption list.')
    if any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f'{video_id}: missing/empty caption.')
    return values


def video_path(root, video_id, activitynet=False):
    candidates = [video_id]
    if activitynet:
        candidates.append(video_id[2:] if video_id.startswith('v_') else 'v_' + video_id)
    for stem in candidates:
        path = root / 'raw_videos' / f'{stem}.mp4'
        if path.is_file():
            return path
    return root / 'raw_videos' / f'{video_id}.mp4'


def load_dataset(name, root, annotations=None, activitynet_mode='segments', activitynet_val_file='val_1.json'):
    root = Path(root)
    if name == 'activitynet':
        name = 'activitynet_captions'
    if name not in DATASETS:
        raise ValueError(f'Unknown dataset: {name}')
    ids, splits = combined_subset(root / 'subset.json')
    if name == 'msrvtt':
        # Retain exact ID and query ordering/configuration of the working MSR-VTT program.
        ids = load_subset(root / 'subset.json')
        if annotations and len(annotations) != 1:
            raise ValueError('MSR-VTT needs exactly one annotation file.')
        path = find_annotations(root, annotations[0] if annotations else None)
        items = [dict(item_id=v, source_video_id=v, path=video_path(root, v), start=None, end=None) for v in ids]
        return Dataset(name, items, load_captions(path, ids), [path], ['train', 'test'], PROTOCOL)
    if activitynet_mode not in ('segments', 'video'):
        raise ValueError('ActivityNet mode must be segments or video.')
    names = ('vatex_training_v1.0.json', 'vatex_validation_v1.0.json') if name == 'vatex' else ('train.json', activitynet_val_file)
    paths = annotation_files(root, names, annotations)
    rows = {}
    for path in paths:
        content = read_json(path)
        if name == 'vatex':
            if not isinstance(content, list):
                raise ValueError(f'{path}: expected VATEX annotation list.')
            pairs = []
            for row in content:
                if not isinstance(row, dict) or not isinstance(row.get('videoID'), str):
                    raise ValueError(f'{path}: VATEX row needs videoID.')
                pairs.append((row['videoID'], row))
        else:
            if not isinstance(content, dict):
                raise ValueError(f'{path}: expected ActivityNet video-keyed object.')
            pairs = content.items()
        for key, row in pairs:
            if not isinstance(row, dict):
                raise ValueError(f'{path}: {key} annotation must be an object.')
            canonical = key if name != 'activitynet_captions' or key.startswith('v_') else 'v_' + key
            if canonical in rows and rows[canonical] != row:
                raise ValueError(f'Conflicting annotations for {canonical}. Use one validation annotation version.')
            rows[canonical] = row
    items, queries = [], []
    seen = set()
    for raw_id in ids:
        v = raw_id if name != 'activitynet_captions' or raw_id.startswith('v_') else 'v_' + raw_id
        if v in seen:
            continue
        seen.add(v)
        if v not in rows:
            raise ValueError(f'No captions/annotations for selected video {v}.')
        row = rows[v]
        path = video_path(root, v, name == 'activitynet_captions')
        captions = caption_list(row.get('enCap') if name == 'vatex' else row.get('sentences'), v)
        if name == 'vatex' or activitynet_mode == 'video':
            items.append(dict(item_id=v, source_video_id=v, path=path, start=None, end=None))
            if name == 'activitynet_captions':
                captions = [' '.join(caption.strip() for caption in captions)]
            queries.extend(dict(query_id=f'{v}:{i}', video_id=v, text=c) for i, c in enumerate(captions))
        else:
            timestamps = row.get('timestamps')
            if not isinstance(timestamps, list) or len(timestamps) != len(captions):
                raise ValueError(f'{v}: timestamps and sentences must have equal lengths.')
            for i, (stamp, caption) in enumerate(zip(timestamps, captions)):
                if not isinstance(stamp, (list, tuple)) or len(stamp) != 2:
                    raise ValueError(f'{v}: invalid timestamps at segment {i}.')
                start, end = map(float, stamp)
                if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                    raise ValueError(f'{v}: invalid timestamps at segment {i}: {stamp}.')
                item_id = f'{v}__seg{i:04d}'
                items.append(dict(item_id=item_id, source_video_id=v, path=path, start=start, end=end))
                queries.append(dict(query_id=item_id, video_id=item_id, source_video_id=v,
                                    start=start, end=end, text=caption))
    protocol = 'combined_subset_all_english_captions' if name == 'vatex' else f'combined_subset_activitynet_{activitynet_mode}'
    return Dataset(name, items, queries, paths, splits, protocol)


from .video_sampling import segment_indices
