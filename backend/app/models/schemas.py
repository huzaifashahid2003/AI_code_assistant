from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel


class ReviewRequest(BaseModel):
    filename: str
    query: str


class UploadResponse(BaseModel):
    filename: str
    saved_to: str
    status: str


class ChunkReview(BaseModel):
    chunk_name: str
    chunk_type: str
    start_line: int
    end_line: int
    source: str
    issue: str
    suggestion: str
    severity: str
    problematic_lines: List[int]


class ReviewResponse(BaseModel):
    filename: str
    query: str
    total_chunks: int
    reviewed_chunks: int
    review: str
    chunk_reviews: List[Any]
    corrected_code: str
    ai_used: bool
    fallback: bool
