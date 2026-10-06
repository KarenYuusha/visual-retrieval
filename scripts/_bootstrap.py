"""Allow repository scripts to run before editable installation and from any cwd."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
