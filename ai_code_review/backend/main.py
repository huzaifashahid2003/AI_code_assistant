import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level above backend/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from code_processing import chunk_code
from analyzer import review_chunk
from rag_search import create_faiss_index, generate_code_review, generate_corrected_code, retrieve_relevant_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="AI Code Review Assistant",
    description="Upload Python files for AI-powered code review.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         
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

    logger.info("File uploaded: '%s' → %s", safe_name, destination)

    return {
        "filename": safe_name,
        "saved_to": str(destination.relative_to(BASE_DIR)),
        "status": "uploaded successfully",
    }

class ReviewRequest(BaseModel):

    filename: str
    query: str


@app.post("/review", tags=["Review"])
def review_file(request: ReviewRequest) -> dict:

    # 1. Resolve path — strip any directory components to prevent traversal
    safe_name = Path(request.filename).name
    file_path = UPLOAD_DIR / safe_name

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"'{safe_name}' not found in uploads. Upload the file first.",
        )

    # 2. Read content
    try:
        code_text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail=f"'{safe_name}' could not be decoded as UTF-8. Ensure the file uses UTF-8 encoding.",
        )

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

    logger.info("'%s' split into %d chunk(s).", safe_name, len(chunks))

    chunk_sources = [
        f"# {c.chunk_type.title()}: {c.name} | Lines {c.start_line}-{c.end_line}\n{c.source}"
        for c in chunks
    ]

    # Map formatted source string → original CodeChunk for later lookup
    source_to_chunk = {src: chunk for src, chunk in zip(chunk_sources, chunks)}

    # 4. Build FAISS index
    store = create_faiss_index(chunk_sources)

    # 5. Retrieve relevant chunks
    relevant = retrieve_relevant_chunks(store, request.query)

    # Resolve relevant source strings back to CodeChunk objects
    relevant_chunk_objs = [
        source_to_chunk[s] for s in relevant if s in source_to_chunk
    ]

    # 6. Generate review
    logger.info("Generating review for '%s' using %d relevant chunk(s).", safe_name, len(relevant))
    review_text = generate_code_review(relevant)
    logger.info("Review generated for '%s'.", safe_name)

    # 7. Per-chunk structured review (issue / suggestion / severity / problematic_lines)
    logger.info("Running per-chunk review for '%s' (%d chunk(s)).", safe_name, len(relevant_chunk_objs))
    chunk_reviews = [review_chunk(c) for c in relevant_chunk_objs]
    logger.info("Per-chunk review complete for '%s'.", safe_name)

    # 8. Generate corrected code
    logger.info("Generating corrected code for '%s'.", safe_name)
    corrected_code = generate_corrected_code(code_text)
    logger.info("Corrected code generated for '%s'.", safe_name)

    return {
        "filename": safe_name,
        "query": request.query,
        "total_chunks": len(chunks),
        "reviewed_chunks": len(relevant),
        "review": review_text,
        "chunk_reviews": chunk_reviews,
        "corrected_code": corrected_code,
    }
