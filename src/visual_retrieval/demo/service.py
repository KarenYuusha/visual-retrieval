"""Namespaced exact search with one serialized, reusable inference encoder."""
import gc
import json
import math
from pathlib import Path
import re
import threading
from zipfile import BadZipFile

import numpy as np

from visual_retrieval.config import MODEL_NAMES
from visual_retrieval.models.registry import validate_preprocessing
from visual_retrieval.retrieval.index import GalleryIndex
from visual_retrieval.retrieval.search import load_saved_encoder


def _space_identity(config, dimension):
    # Conservatively compare full provenance (including checkpoint path). A
    # copied checkpoint with a different signature requires consistent extraction.
    fields = ('model_id', 'source_commit', 'preprocessing', 'tokenizer', 'checkpoint', 'precision')
    missing = [key for key in fields if key not in config]
    if missing:
        raise ValueError(f'Missing encoder provenance: {missing}')
    return json.dumps({**{key: config[key] for key in fields}, 'dimension': dimension}, sort_keys=True)


def _item_metadata(spec, item_id, item, segmented):
    if item is None:
        if segmented:
            raise ValueError(f'Segment metadata missing: {item_id}')
        item = dict(source_video_id=item_id, path=f'{item_id}.mp4', start=None, end=None)
    start, end = item.get('start'), item.get('end')
    if (segmented and start is None) or (start is None) != (end is None):
        raise ValueError(f'Both segment bounds are required: {item_id}')
    if start is not None and (type(start) not in (int, float) or type(end) not in (int, float)
            or not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start):
        raise ValueError(f'Invalid segment interval: {item_id}')
    source = Path(str(item['path']).replace('\\', '/'))
    if '..' in source.parts or not re.fullmatch(r'[A-Za-z0-9_.-]+\.mp4', source.name, re.IGNORECASE):
        raise ValueError(f'Invalid video filename: {item_id}')
    video_root = (spec.root / 'raw_videos').resolve()
    path = (video_root / source.name).resolve()
    if not path.is_relative_to(video_root):
        raise ValueError(f'Video path leaves the configured root: {item_id}')
    return dict(key=json.dumps([spec.key, item_id]), dataset=spec.key, dataset_label=spec.label,
                item_id=item_id, source_video_id=item.get('source_video_id', item_id),
                path=str(path), start=start, end=end)


class SearchService:
    def __init__(self, datasets, device='cuda', encoder_loader=load_saved_encoder):
        self.device, self.encoder_loader = device, encoder_loader
        self.galleries, self.issues, self._indexes = {}, [], {}
        self._encoder, self._encoder_key = None, None
        self._lock = threading.RLock()
        keys = [spec.key for spec in datasets]
        if len(keys) != len(set(keys)):
            raise ValueError('Dataset keys must be unique.')
        for spec in datasets:
            for model, folder in spec.feature_dirs.items():
                folder = Path(folder)
                try:
                    config = json.loads((folder / 'run_config.json').read_text(encoding='utf-8'))
                    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
                    if not isinstance(config, dict) or not isinstance(manifest, dict):
                        raise ValueError('Saved manifests must be JSON objects.')
                    if validate_preprocessing(config) != model or validate_preprocessing(manifest) != model:
                        raise ValueError('Saved model does not match configured gallery.')
                    # Only load video arrays: query embeddings are not needed by the demo.
                    with np.load(folder / 'features.npz', allow_pickle=False) as archive:
                        if archive['video_ids'].ndim != 1:
                            raise ValueError('Gallery IDs must be a 1D array.')
                        ids, features = archive['video_ids'].tolist(), archive['video_features']
                    if not all(isinstance(item, str) and item for item in ids):
                        raise ValueError('Gallery IDs must be nonempty strings.')
                    index = GalleryIndex(ids, features)
                    identity = _space_identity(config, features.shape[1])
                    if _space_identity(manifest, features.shape[1]) != identity:
                        raise ValueError('Manifest and run_config encoder provenance differ.')
                    effective = manifest.get('effective_precision')
                    if effective not in (None, 'fp32', 'fp16', 'bf16'):
                        raise ValueError('Invalid effective extraction precision.')
                    # Legacy/all-cache runs may omit effective precision. Let such
                    # galleries share the recorded encoder settings; do not invent
                    # a historical dtype or mix unknown and recorded precision.
                    identity += json.dumps({'effective_precision': effective}, sort_keys=True)
                    requested = manifest['requested_video_ids']
                    if set(ids) - set(requested):
                        raise ValueError('Gallery includes unrequested items.')
                    rows = manifest.get('gallery_items', [])
                    if not isinstance(rows, list) or not all(isinstance(x, dict) for x in rows):
                        raise ValueError('Gallery metadata must be a list of objects.')
                    if len({x['item_id'] for x in rows}) != len(rows):
                        raise ValueError('Duplicate gallery metadata IDs.')
                    items = {x['item_id']: x for x in rows}
                    segmented = manifest.get('protocol') == 'combined_subset_activitynet_segments'
                    metadata = [_item_metadata(spec, item, items.get(item), segmented) for item in ids]
                    self.galleries[(model, spec.key)] = dict(spec=spec, folder=folder, index=index,
                        identity=identity, effective_precision=effective, metadata=metadata, omitted=len(set(requested) - set(ids)))
                except (ValueError, OSError, KeyError, TypeError, EOFError, BadZipFile) as error:
                    self.issues.append(f'{spec.label} / {model}: {error}')
        if not self.galleries:
            raise ValueError('No searchable galleries found. Extract features first.\n' + '\n'.join(self.issues))

    @property
    def available_models(self):
        return [model for model in MODEL_NAMES if any(m == model for m, _ in self.galleries)]

    def dataset_choices(self, model):
        return [('All datasets', 'all')] + [(g['spec'].label, dataset) for (m, dataset), g in self.galleries.items() if m == model]

    def gallery_counts(self, model):
        return {dataset: len(g['metadata']) for (m, dataset), g in self.galleries.items() if m == model}

    def omitted_counts(self, model):
        return {dataset: g['omitted'] for (m, dataset), g in self.galleries.items() if m == model}

    def _selected(self, model, dataset):
        galleries = [g for (m, key), g in self.galleries.items() if m == model and (dataset == 'all' or key == dataset)]
        if not galleries:
            raise ValueError('No saved gallery is available for this model and dataset.')
        if len({g['identity'] for g in galleries}) != 1:
            raise ValueError('Combined search requires compatible checkpoint, tokenizer, precision and preprocessing. Select one dataset or re-extract matching galleries.')
        return galleries

    def _ensure_encoder(self, gallery):
        key = gallery['identity']
        if self._encoder_key != key:
            self.close()
            self._encoder = self.encoder_loader(gallery['folder'], self.device)
            expected = gallery['effective_precision']
            if self.device != 'cpu' and expected is not None and self._encoder.effective_precision != expected:
                self.close()
                raise ValueError('Query precision differs from extraction. Use matching GPU precision or re-extract the gallery.')
            self._encoder_key = key
        return self._encoder

    def warmup(self, model):
        with self._lock:
            # Warm a valid individual gallery even if a combined space is incompatible.
            gallery = next((g for (m, _), g in self.galleries.items() if m == model), None)
            if gallery is None:
                raise ValueError('No saved gallery is available for this model.')
            self._ensure_encoder(gallery).encode_text(['a video'])

    def search(self, text, model, dataset='all', top_k=10):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Enter a video description.')
        if len(text) > 2000 or type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError('Use a description up to 2000 characters and top_k from 1 to 100.')
        with self._lock:
            galleries = self._selected(model, dataset)
            index_key = (model, dataset)
            if index_key not in self._indexes:
                metadata = [item for g in galleries for item in g['metadata']]
                features = np.concatenate([g['index'].features for g in galleries])
                index = GalleryIndex([item['key'] for item in metadata], features)
                self._indexes[index_key] = (index, {item['key']: item for item in metadata})
            index, metadata = self._indexes[index_key]
            query = self._ensure_encoder(galleries[0]).encode_text([text.strip()])[0]
            results = index.search(query, top_k)
            return [dict(**metadata[row['item_id']], model=model, rank=row['rank'], cosine_similarity=row['cosine_similarity']) for row in results]

    def resolve_result(self, row):
        """Resolve selected media from the registry, never from a client path."""
        gallery = self.galleries.get((row.get('model'), row.get('dataset')))
        if gallery is None or row.get('item_id') not in gallery['index'].lookup:
            raise ValueError('Select a result from the current search.')
        return gallery['metadata'][gallery['index'].lookup[row['item_id']]]

    def close(self):
        with self._lock:
            had_encoder = self._encoder is not None
            self._encoder, self._encoder_key = None, None
            if had_encoder:
                gc.collect()
                if self.device == 'cuda':
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
