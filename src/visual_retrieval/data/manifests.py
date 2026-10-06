"""Canonical in-memory dataset manifest."""
from dataclasses import dataclass

@dataclass
class Dataset:
    name: str
    items: list
    queries: list
    annotation_paths: list
    splits: list
    protocol: str
