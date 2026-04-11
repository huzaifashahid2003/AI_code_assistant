import sys
from pathlib import Path

# Allow `from app.xxx import ...` when pytest is run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
