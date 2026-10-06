"""Legacy HDF5 compatibility entrypoint. Prefer scripts/ for new comparisons."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from visual_retrieval.legacy.extract_clip_features import *
if __name__ == '__main__':
    main()
