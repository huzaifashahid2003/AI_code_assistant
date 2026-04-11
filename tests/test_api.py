"""
test_api.py — Integration tests for the AI Code Review Assistant API.

Run from the project root (ai_code_review/):
    pytest tests/
"""
import io

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_returns_ok():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "message" in data


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def test_upload_rejects_non_py_extension():
    response = client.post(
        "/upload",
        files={"file": ("script.txt", b"print('hello')", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_rejects_missing_extension():
    response = client.post(
        "/upload",
        files={"file": ("noextension", b"x = 1", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_accepts_valid_py_file():
    code = b"def foo():\n    pass\n"
    response = client.post(
        "/upload",
        files={"file": ("valid_upload.py", code, "text/x-python")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "valid_upload.py"
    assert data["status"] == "uploaded successfully"


def test_upload_rejects_oversized_file():
    big_content = b"# x\n" * 60_000  # well over 200 KB
    response = client.post(
        "/upload",
        files={"file": ("big.py", big_content, "text/x-python")},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Review — validation
# ---------------------------------------------------------------------------

def test_review_rejects_empty_query():
    response = client.post(
        "/review",
        json={"filename": "valid_upload.py", "query": ""},
    )
    assert response.status_code == 400


def test_review_rejects_whitespace_only_query():
    response = client.post(
        "/review",
        json={"filename": "valid_upload.py", "query": "   "},
    )
    assert response.status_code == 400


def test_review_returns_404_for_missing_file():
    response = client.post(
        "/review",
        json={"filename": "does_not_exist.py", "query": "find bugs"},
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Review — successful flow
# ---------------------------------------------------------------------------

def test_review_returns_expected_fields():
    """Upload a small file then review it; verify the response shape."""
    code = b"def add(a, b):\n    return a + b\n\ndef divide(a, b):\n    return a / b\n"

    # Upload
    upload_resp = client.post(
        "/upload",
        files={"file": ("review_test.py", code, "text/x-python")},
    )
    assert upload_resp.status_code == 200

    # Review
    review_resp = client.post(
        "/review",
        json={"filename": "review_test.py", "query": "check for division by zero"},
    )
    assert review_resp.status_code == 200

    data = review_resp.json()
    assert "review" in data
    assert "chunk_reviews" in data
    assert "corrected_code" in data
    assert "total_chunks" in data
    assert "reviewed_chunks" in data


def test_review_response_contains_ai_flags():
    """ai_used and fallback must be present and mutually exclusive."""
    code = b"def hello():\n    print('hi')\n"

    client.post("/upload", files={"file": ("flags_test.py", code, "text/x-python")})

    response = client.post(
        "/review",
        json={"filename": "flags_test.py", "query": "review this"},
    )
    assert response.status_code == 200

    data = response.json()
    assert "ai_used" in data
    assert "fallback" in data
    # Exactly one of the two must be True
    assert data["ai_used"] != data["fallback"]
