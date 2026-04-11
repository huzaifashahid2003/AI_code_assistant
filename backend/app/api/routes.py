"""
routes.py — FastAPI route handlers.

Applies input validation before any processing so failures are reported
immediately with a meaningful HTTP status and message.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import MAX_FILE_SIZE, UPLOAD_DIR
from app.models.schemas import ReviewRequest, ReviewResponse, UploadResponse
from app.services.correction_service import generate_corrected_code
from app.services.rag_service import build_tfidf_index, retrieve_relevant_chunks
from app.services.review_service import generate_code_review, review_chunk
from app.utils.code_processing import chunk_code

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", tags=["Health"])
def health_check() -> dict:
    """Simple liveness probe."""
    return {"status": "ok", "message": "AI Code Review Assistant is running."}


@router.post("/upload", tags=["Upload"])
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a .py file, validate it, persist it, and return metadata."""
    if not file.filename or not file.filename.endswith(".py"):
        raise HTTPException(
            status_code=400,
            detail="Only Python (.py) files are accepted.",
        )

    safe_name = Path(file.filename).name
    destination = UPLOAD_DIR / safe_name

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large — maximum {MAX_FILE_SIZE:,} bytes (~200 KB).",
        )

    destination.write_bytes(contents)
    logger.info("File uploaded: '%s' → %s", safe_name, destination)

    # Compute path relative to project root for the response
    base_dir = UPLOAD_DIR.parent.parent  # data/ → ai_code_review/
    return UploadResponse(
        filename=safe_name,
        saved_to=str(destination.relative_to(base_dir)),
        status="uploaded successfully",
    )


@router.post("/review", tags=["Review"])
def review_file(request: ReviewRequest) -> ReviewResponse:
    """Chunk, retrieve, review, and correct the uploaded Python file."""

    # ── Input validation ──────────────────────────────────────────────────
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    safe_name = Path(request.filename).name
    file_path = UPLOAD_DIR / safe_name

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"'{safe_name}' not found in uploads. Upload the file first.",
        )

    # ── Read source ───────────────────────────────────────────────────────
    try:
        code_text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail=f"'{safe_name}' could not be decoded as UTF-8.",
        )

    if not code_text.strip():
        raise HTTPException(
            status_code=400,
            detail=f"'{safe_name}' is empty — nothing to review.",
        )

    if len(code_text.encode()) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large — maximum {MAX_FILE_SIZE:,} bytes.",
        )

    # ── Chunk ─────────────────────────────────────────────────────────────
    chunks = chunk_code(code_text)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No top-level functions or classes found in '{safe_name}'. "
                "Ensure the file contains at least one def or class statement."
            ),
        )

    logger.info("'%s' split into %d chunk(s).", safe_name, len(chunks))

    chunk_sources = [
        f"# {c.chunk_type.title()}: {c.name} | Lines {c.start_line}-{c.end_line}\n{c.source}"
        for c in chunks
    ]

    # ── Retrieve — index-based, no fragile string mapping ─────────────────
    store = build_tfidf_index(chunk_sources)
    indices = retrieve_relevant_chunks(store, request.query)

    relevant_sources = [chunk_sources[i] for i in indices]
    relevant_chunk_objs = [chunks[i] for i in indices]

    # ── Review ────────────────────────────────────────────────────────────
    logger.info(
        "Generating review for '%s' using %d relevant chunk(s).",
        safe_name, len(relevant_sources),
    )
    review_text = generate_code_review(relevant_sources)
    chunk_reviews = [review_chunk(c) for c in relevant_chunk_objs]

    # ── Correct ───────────────────────────────────────────────────────────
    logger.info("Generating corrected code for '%s'.", safe_name)
    corrected_code = generate_corrected_code(code_text)

    # ── AI availability flag ───────────────────────────────────────────────
    ai_used = bool(os.environ.get("GROQ_API_KEY"))

    logger.info("Review complete for '%s'.", safe_name)

    return ReviewResponse(
        filename=safe_name,
        query=request.query,
        total_chunks=len(chunks),
        reviewed_chunks=len(relevant_sources),
        review=review_text,
        chunk_reviews=chunk_reviews,
        corrected_code=corrected_code,
        ai_used=ai_used,
        fallback=not ai_used,
    )
