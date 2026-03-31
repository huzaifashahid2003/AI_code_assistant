"""
main.py — FastAPI server for AI Code Review Assistant.

Endpoints:
    POST /upload  — Accept a Python .py file, save it, and return metadata.
"""

import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Code Review Assistant",
    description="Upload Python files for AI-powered code review.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root directory of *this* file → project root is two levels up
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health_check() -> dict:
    """Simple liveness probe."""
    return {"status": "ok", "message": "AI Code Review Assistant is running."}


@app.post("/upload", tags=["Upload"])
async def upload_file(file: UploadFile = File(...)) -> dict:
    """
    Accept a Python source file (.py), persist it under data/uploads/, and
    return basic metadata.

    Args:
        file: The uploaded file (must have a .py extension).

    Returns:
        JSON with filename, saved path, and status.

    Raises:
        HTTPException 400: If the file is not a .py file.
    """
    # Validate extension
    if not file.filename or not file.filename.endswith(".py"):
        raise HTTPException(
            status_code=400,
            detail="Only Python (.py) files are accepted.",
        )

    # Sanitise the filename to prevent path-traversal attacks
    safe_name = Path(file.filename).name
    destination = UPLOAD_DIR / safe_name

    # Stream file to disk
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": safe_name,
        "saved_to": str(destination.relative_to(BASE_DIR)),
        "status": "uploaded successfully",
    }
