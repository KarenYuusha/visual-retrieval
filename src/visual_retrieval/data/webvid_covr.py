"""Aligned image/edit/target examples from the WebVid-CoVR subset downloader."""
from dataclasses import dataclass
import json
from pathlib import Path
import re

from visual_retrieval.common import sha256

PROTOCOL = 'webvid_covr_image_edit_subset_runtime_v1'


@dataclass
class WebVidSubset:
    root: Path
    split: str
    subset_path: Path
    subset_sha256: str
    items: list
    queries: list
    requested_samples: int
    downloaded_samples: int
    download_complete: bool
    download_failures: list

    def candidate_ids(self, mode='targets'):
        if mode == 'all':
            return [item['item_id'] for item in self.items]
        if mode != 'targets':
            raise ValueError('Gallery must be targets or all.')
        return list(dict.fromkeys(query['video_id'] for query in self.queries))


def _path(root, value):
    if not isinstance(value, str) or not value or ':' in value or '\\' in value:
        raise ValueError('Expected a relative path inside the dataset root.')
    path = Path(value)
    resolved = (root / path).resolve()
    if path.is_absolute() or '..' in path.parts or not resolved.is_relative_to(root):
        raise ValueError('Dataset path leaves its root.')
    return resolved


def load_subset(root, split='test', subset_path=None, limit=None, allow_incomplete=False):
    root = Path(root).expanduser().resolve()
    if split not in ('train', 'val', 'test') or (limit is not None and (type(limit) is not int or limit < 1)):
        raise ValueError('Use a known split and a positive limit.')
    path = Path(subset_path or root / 'samples' / split / 'subset.json').resolve()
    doc = json.loads(path.read_text(encoding='utf-8-sig'))
    if doc.get('dataset') != 'WebVid-CoVR' or doc.get('split') != split:
        raise ValueError('Expected the downloader WebVid-CoVR subset for this split.')
    if type(doc.get('complete')) is not bool:
        raise ValueError('Expected downloader completeness metadata.')
    if not doc['complete'] and not allow_incomplete:
        raise ValueError('Download subset is incomplete; finish it or use --allow-incomplete explicitly.')
    samples = doc.get('samples')
    if not isinstance(samples, list) or not samples or not all(isinstance(row, dict) for row in samples):
        raise ValueError('The subset must have complete sample rows.')
    requested=doc.get('requested_samples')
    if (type(requested) is not int or requested<len(samples)
            or (doc['complete'] and requested!=len(samples))):
        raise ValueError('Downloader sample counts are inconsistent.')
    failures=doc.get('failures',[])
    if not isinstance(failures,list):raise ValueError('Invalid downloader failure metadata.')
    selected = samples[:limit] if limit is not None else samples
    items, queries, seen = {}, [], set()
    for row in selected:
        index, reference, target = row.get('row_index'), row.get('reference_id'), row.get('target_id')
        if type(index) is not int or index < 0:
            raise ValueError('Expected a nonnegative annotation row_index.')
        if index in seen:
            raise ValueError('Duplicate query row_index.')
        seen.add(index)
        if not all(isinstance(id, str) and re.fullmatch(r'[0-9]+', id) for id in (reference,target)):
            raise ValueError('Expected numeric WebVid video IDs.')
        if reference == target:
            raise ValueError('A reference cannot also be its own target.')
        text = row.get('modification')
        if not isinstance(text,str) or not text.strip():
            raise ValueError('Each sample requires modification text.')
        for id, field in ((reference,'reference_video'),(target,'target_video')):
            video = _path(root,row.get(field))
            if video != (root / 'raw_videos' / f'{id}.mp4').resolve():
                raise ValueError('Video path and ID are not aligned with raw_videos.')
            item = dict(item_id=id,source_video_id=id,path=video,start=None,end=None)
            if id in items and items[id] != item:
                raise ValueError('Conflicting metadata for the same video ID.')
            items[id] = item
        image=_path(root,row.get('reference_image'))
        if image != root/'reference_frames'/f'{reference}.jpg':
            raise ValueError('Reference image does not match its reference video ID.')
        queries.append(dict(query_id=f'{split}_row_{index}',row_index=index,reference_id=reference,
                            video_id=target,text=text.strip(),reference_image=image))
    return WebVidSubset(root,split,path,sha256(path),list(items.values()),queries,
                        requested,len(samples),doc['complete'],failures)


def check_media(data):
    missing = [str(item['path']) for item in data.items if not item['path'].is_file()]
    missing += [str(q['reference_image']) for q in data.queries if not q['reference_image'].is_file()]
    return sorted(set(missing))


def serializable_queries(queries):
    return [dict(query,reference_image=str(query['reference_image'])) for query in queries]
