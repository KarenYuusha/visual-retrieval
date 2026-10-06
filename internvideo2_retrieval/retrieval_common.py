"""Compatibility imports from visual_retrieval.common."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from visual_retrieval.common import *
