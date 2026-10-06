"""Compatibility entrypoint; implementation lives in visual_retrieval.retrieval.search."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from visual_retrieval.retrieval.search import *

if __name__ == '__main__':
    main()
