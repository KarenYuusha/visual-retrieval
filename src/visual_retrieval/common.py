"""Dataset handling shared by extraction, evaluation, and search."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

DEFAULT_ROOT = r'S:\video_retrieval\msr_vtt'
PROTOCOL = 'combined_train_test_all_captions'
PREPROCESSING_V1 = '4_middle_frames_uint8_bicubic224_imagenet_text_clean_max40_v1'
PREPROCESSING_V2 = '4_middle_frames_uint8_bicubic224_imagenet_text_clean_cls_only_max40_v2'


def load_subset(path):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    ids = []
    for split in ('train', 'test'):
        if not isinstance(data.get(split), list):
            raise ValueError(f'{path}: expected a list under {split!r}.')
        for value in data[split]:
            if not isinstance(value, str):
                raise ValueError(f'{split}: expected video ID strings, got {value!r}.')
            video_id = value.removesuffix('.mp4')
            if not re.fullmatch(r'video\d+', video_id):
                raise ValueError(f'Invalid MSR-VTT video ID: {value!r}.')
            if video_id not in ids:
                ids.append(video_id)
    if not ids:
        raise ValueError('The combined train+test subset is empty.')
    return ids


def load_captions(path, video_ids):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    rows = data.get('sentences') if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f'{path}: expected MSR-VTT annotations with a sentences list.')
    selected = set(video_ids)
    queries = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'Annotation row {index} is not an object.')
        video_id = row.get('video_id')
        if video_id not in selected:
            continue
        caption = row.get('caption')
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f'Missing/empty caption for {video_id}, row {index}.')
        queries.append({'query_id': row.get('sen_id', index),
                        'video_id': video_id, 'text': caption})
    missing = selected - {q['video_id'] for q in queries}
    if missing:
        raise ValueError('No captions for selected videos: ' + ', '.join(sorted(missing)[:20]))
    return queries


def find_annotations(root, explicit=None):
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for relative in ('raw_data/MSRVTT_data.json', 'MSRVTT_data.json'):
        path = Path(root) / relative
        if path.is_file():
            return path
    raise FileNotFoundError('Captions are required. Expected raw_data/MSRVTT_data.json '
                            'under --data-root, or supply --annotations PATH.')


def clean_text(text):
    # Matches multi_modality/dataset/utils.py:pre_text in the official repository.
    text = re.sub(r'([,.\'!?"()*#:;~])', '', text.lower())
    text = text.replace('-', ' ').replace('/', ' ').replace('<person>', 'person')
    return re.sub(r'\s{2,}', ' ', text).rstrip('\n').strip(' ')


def middle_indices(length, count=4):
    if length <= 0 or count <= 0:
        raise ValueError('Video length and sample count must be positive.')
    n = min(length, count)
    edges = np.linspace(0, length, n + 1).astype(int)
    indices = [(int(edges[i]) + int(edges[i + 1]) - 1) // 2 for i in range(n)]
    return indices + [indices[-1]] * (count - n)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    return {'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_features(features):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
        raise ValueError('Features must be a nonempty, finite 2D matrix.')
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError('Zero-norm embeddings found.')
    return features / norms


def read_feature_bundle(directory):
    directory = Path(directory)
    with np.load(directory / 'features.npz', allow_pickle=False) as archive:
        bundle = {key: archive[key] for key in archive.files}
    bundle['video_features'] = normalize_features(bundle['video_features'])
    bundle['text_features'] = normalize_features(bundle['text_features'])
    vids = bundle['video_ids'].tolist()
    targets = bundle['text_video_ids'].tolist()
    if len(vids) != len(set(vids)) or len(vids) != len(bundle['video_features']):
        raise ValueError('Gallery IDs must be unique and match the feature rows.')
    if len(targets) != len(bundle['text_features']) or set(targets) - set(vids):
        raise ValueError('Text labels do not match gallery IDs.')
    if bundle['text_features'].shape[1] != bundle['video_features'].shape[1]:
        raise ValueError('Text and video feature dimensions differ.')
    return bundle
