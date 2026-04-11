import os
from pathlib import Path

from dotenv import load_dotenv

# Project root is four levels above this file:
#   config.py → core/ → app/ → backend/ → ai_code_review/
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"
TFIDF_MAX_FEATURES: int = 512
MAX_FILE_SIZE: int = 200_000  # ~200 KB upper bound for uploaded source files

UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
