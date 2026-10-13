"""Configurable dataset galleries, including future prepared datasets."""
from dataclasses import dataclass
from pathlib import Path
import re
import yaml

from visual_retrieval.config import DEFAULT_BASE, DATASET_NAMES, MODEL_NAMES, MODEL_OUTPUTS, dataset_root


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    root: Path
    feature_dirs: dict


@dataclass(frozen=True)
class DemoConfig:
    datasets: list
    default_model: str = 'internvideo2'
    top_k: int = 10

    @classmethod
    def defaults(cls, base=DEFAULT_BASE):
        labels = {'msrvtt': 'MSR-VTT', 'vatex': 'VATEX', 'activitynet': 'ActivityNet'}
        return cls([DatasetSpec(key, labels[key], dataset_root(key, base),
                    {model: dataset_root(key, base) / 'features' / folder for model, folder in MODEL_OUTPUTS.items()})
                    for key in DATASET_NAMES])

    @classmethod
    def from_yaml(cls, path):
        path = Path(path).resolve()
        with path.open(encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict) or not isinstance(config.get('datasets'), dict) or not config['datasets']:
            raise ValueError('Demo configuration requires a nonempty datasets mapping.')
        if set(config) - {'datasets', 'default_model', 'top_k'}:
            raise ValueError('Unknown demo configuration fields.')
        datasets = []
        for key, row in config['datasets'].items():
            if not isinstance(key, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', key) or key == 'all':
                raise ValueError(f'Invalid or reserved dataset key: {key}')
            if not isinstance(row, dict) or 'root' not in row or set(row) - {'root', 'label', 'features'}:
                raise ValueError(f'{key}: expected root, optional label and features.')
            root = Path(row['root']).expanduser()
            root = root if root.is_absolute() else path.parent / root
            root = root.resolve()
            features = row.get('features', {m: f'features/{folder}' for m, folder in MODEL_OUTPUTS.items()})
            if not isinstance(features, dict) or not features or set(features) - set(MODEL_NAMES):
                raise ValueError(f'{key}: features must map known model names to directories.')
            folders = {}
            for model, folder in features.items():
                folder = Path(folder).expanduser()
                folders[model] = (folder if folder.is_absolute() else root / folder).resolve()
            label = row.get('label', key)
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f'{key}: label must be nonempty.')
            datasets.append(DatasetSpec(key, label, root, folders))
        model, top_k = config.get('default_model', 'internvideo2'), config.get('top_k', 10)
        if model not in MODEL_NAMES or type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError('Invalid default_model or top_k (expected 1 to 100).')
        return cls(datasets, model, top_k)
