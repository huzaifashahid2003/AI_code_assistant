from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Ensure .env is loaded even when this module is imported before main.py runs
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

try:
    from groq import Groq as _GroqClass  # type: ignore[import]
    _GROQ_AVAILABLE = True
except ImportError:
    _GroqClass = None
    _GROQ_AVAILABLE = False

try:
    import faiss  # noqa: F401
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

GROQ_MODEL = "llama-3.3-70b-versatile"


def _get_groq_client():
    """Lazy Groq client — created at call time so the key is always current."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or not _GROQ_AVAILABLE:
        return None
    return _GroqClass(api_key=api_key)

_CHUNK_REVIEW_PROMPT = """\
You are an expert Python code reviewer. Review the following Python code chunk \
and return a JSON object.

IMPORTANT: Line numbering in this chunk starts at line {start_line} (not line 1). \
The chunk spans lines {start_line} to {end_line} in the original file.

Python code chunk:
```python
{source}
```

Return ONLY a valid JSON object with EXACTLY these four keys:
- "issue": A concise description of the main issue found (string). \
  Write "No issue found." if the code is clean.
- "suggestion": A concrete, actionable fix (string). Write "None." if no issue.
- "severity": One of "high", "medium", "low", or "none" (string).
- "problematic_lines": A list of ABSOLUTE line numbers (integers) from the \
  original file that contain the issue. Use [] if there is no issue. \
  All numbers MUST be between {start_line} and {end_line} inclusive.

Example output:
{{"issue": "Division by zero not handled", "suggestion": "Add a guard: if count == 0: return 0", "severity": "high", "problematic_lines": [23, 24]}}

Return ONLY the JSON object. No explanation, no markdown fences, no extra text."""


def review_chunk(chunk) -> dict:
    """Review a single CodeChunk and return structured per-chunk feedback.

    Args:
        chunk: A CodeChunk dataclass instance with .name, .chunk_type,
               .source, .start_line, .end_line fields.

    Returns:
        dict with keys:
          chunk_name, chunk_type, start_line, end_line, source,
          issue, suggestion, severity, problematic_lines
    """
    result = {
        "chunk_name": chunk.name,
        "chunk_type": chunk.chunk_type,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "source": chunk.source,
        "issue": "No issue found.",
        "suggestion": "None.",
        "severity": "none",
        "problematic_lines": [],
    }

    client = _get_groq_client()
    if client is None:
        result["issue"] = (
            "Groq unavailable — set GROQ_API_KEY and install groq."
        )
        return result

    prompt = _CHUNK_REVIEW_PROMPT.format(
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        source=chunk.source,
    )

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        raw = completion.choices[0].message.content or ""

        # Strip accidental markdown fences the model may still emit
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())

        parsed = json.loads(raw)

        result["issue"] = str(parsed.get("issue", "No issue found."))
        result["suggestion"] = str(parsed.get("suggestion", "None."))

        severity = str(parsed.get("severity", "none")).lower()
        result["severity"] = (
            severity if severity in {"high", "medium", "low", "none"} else "none"
        )

        raw_lines = parsed.get("problematic_lines", [])
        # Keep only integers that fall within the valid line range
        result["problematic_lines"] = [
            int(ln)
            for ln in raw_lines
            if str(ln).lstrip("-").isdigit()
            and chunk.start_line <= int(ln) <= chunk.end_line
        ]

    except json.JSONDecodeError as exc:
        logger.error("review_chunk JSON parse failed for '%s': %s", chunk.name, exc)
        result["issue"] = f"Could not parse Gemini response: {exc}"
    except Exception as exc:  # noqa: BLE001
        logger.error("review_chunk failed for '%s': %s", chunk.name, exc)
        result["issue"] = f"Review call failed: {exc}"

    return result
