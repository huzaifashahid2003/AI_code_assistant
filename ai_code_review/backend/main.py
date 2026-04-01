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
from pydantic import BaseModel

from code_processing import chunk_code
from rag_search import create_faiss_index, generate_code_review, retrieve_relevant_chunks

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


# ---------------------------------------------------------------------------
# Review endpoint
# ---------------------------------------------------------------------------

class ReviewRequest(BaseModel):
    """
    Request body for the /review endpoint.

    Attributes:
        filename: Name of a previously uploaded .py file (basename only).
        query:    Free-text question or review focus for the AI.
    """

    filename: str
    query: str


@app.post("/review", tags=["Review"])
def review_file(request: ReviewRequest) -> dict:
    """
    Run an AI code review on an uploaded Python file.

    Steps:
        1. Resolve and validate the file path.
        2. Read and validate file content.
        3. Split into AST-based code chunks via ``chunk_code``.
        4. Build a FAISS vector index over chunk embeddings.
        5. Retrieve the most relevant chunks for the user query.
        6. Generate a Markdown review via the configured LLM.

    Args:
        request: JSON body with ``filename`` and ``query``.

    Returns:
        JSON with filename, query, chunk counts, and the review text.

    Raises:
        HTTPException 400: Empty file or no parseable chunks found.
        HTTPException 404: File not found in the uploads directory.
    """
    # 1. Resolve path — strip any directory components to prevent traversal
    safe_name = Path(request.filename).name
    file_path = UPLOAD_DIR / safe_name

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"'{safe_name}' not found in uploads. Upload the file first.",
        )

    # 2. Read content
    code_text = file_path.read_text(encoding="utf-8")
    if not code_text.strip():
        raise HTTPException(
            status_code=400,
            detail=f"'{safe_name}' is empty — nothing to review.",
        )

    # 3. Chunk
    chunks = chunk_code(code_text)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No top-level functions or classes found in '{safe_name}'. "
                "Ensure the file contains at least one def or class statement."
            ),
        )

    chunk_sources = [c.source for c in chunks]

    # 4. Build FAISS index
    store = create_faiss_index(chunk_sources)

    # 5. Retrieve relevant chunks
    relevant = retrieve_relevant_chunks(store, request.query)

    # 6. Generate review
    review_text = generate_code_review(relevant)

    return {
        "filename": safe_name,
        "query": request.query,
        "total_chunks": len(chunks),
        "reviewed_chunks": len(relevant),
        "review": review_text,
    }
