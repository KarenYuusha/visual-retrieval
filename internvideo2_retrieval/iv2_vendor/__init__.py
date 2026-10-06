"""Compatibility namespace for the licensed upstream modules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from visual_retrieval.models import iv2_vendor as _vendor
__path__ = _vendor.__path__
