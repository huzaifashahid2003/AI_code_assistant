"""
review_service.py — Per-chunk structured review and full-file narrative review.

Both functions call Groq when a key is available; they degrade gracefully to
a local placeholder when Groq is absent or the call fails.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import List

from app.core.config import GROQ_MODEL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Groq client helpers
# ---------------------------------------------------------------------------

def _get_groq_client():
    """Lazy Groq client — created at call time so the key is always current."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq  # type: ignore[import]
        return Groq(api_key=api_key)
    except ImportError:
        return None


def _call_groq(client, prompt: str, temperature: float = 0.2) -> str:
    """Send *prompt* to Groq and return the text response."""
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    text = completion.choices[0].message.content
    if not text:
        raise RuntimeError("Groq returned an empty response.")
    return text


# ---------------------------------------------------------------------------
# Per-chunk review
# ---------------------------------------------------------------------------

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
    """Return structured per-chunk feedback for a single CodeChunk."""
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
        result["issue"] = "Groq unavailable — set GROQ_API_KEY and install groq."
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
        result["problematic_lines"] = [
            int(ln)
            for ln in raw_lines
            if str(ln).lstrip("-").isdigit()
            and chunk.start_line <= int(ln) <= chunk.end_line
        ]

    except json.JSONDecodeError as exc:
        logger.error("review_chunk JSON parse failed for '%s': %s", chunk.name, exc)
        result["issue"] = f"Could not parse Groq response: {exc}"
    except Exception as exc:  # noqa: BLE001
        logger.error("review_chunk failed for '%s': %s", chunk.name, exc)
        result["issue"] = f"Review call failed: {exc}"

    return result


# ---------------------------------------------------------------------------
# Full-file narrative review
# ---------------------------------------------------------------------------

_REVIEW_PROMPT = (
    "You are an expert Python code reviewer conducting a professional code review.\n"
    "Analyse the following Python code chunks and produce a structured report.\n\n"
    "For each issue, reference the exact function or class name and the approximate\n"
    "line numbers provided in the chunk headers.\n\n"
    "Structure your entire response using EXACTLY these four sections in this order.\n"
    "Only include a section if you have findings for it -- otherwise omit it.\n\n"
    "---\n\n"
    "## 1. Code Quality Issues\n"
    "List issues related to readability, PEP 8 violations, overly complex logic,\n"
    "missing docstrings, or poor code organisation.\n\n"
    "## 2. Performance Improvements\n"
    "Highlight inefficiencies, unnecessary computations, suboptimal data structures,\n"
    "or operations that could be vectorised or cached.\n\n"
    "## 3. Naming & Style Suggestions\n"
    "Point out unclear variable names, non-snake_case identifiers, magic numbers,\n"
    "or style inconsistencies that reduce readability.\n\n"
    "## 4. Potential Bugs\n"
    "Identify logic errors, unhandled edge cases, missing input validation, or "
    "other likely runtime failures.\n\n"
    "---\n\n"
    "For every finding use this format:\n"
    "- **`FunctionOrClassName` (Lines X-Y):** Brief description of the issue.\n"
    "  *Suggestion:* Concrete, actionable fix.\n\n"
    "If the code is clean in a category, write: *(No issues found.)*\n\n"
    "--- CODE ---\n"
    "{code_context}\n"
    "------------\n\n"
    "Be concise, professional, and constructive. Respond in well-formatted Markdown."
)


def _placeholder_review(chunks: List[str]) -> str:
    chunk_list = "".join(
        f"- **Chunk {i}:** `{chunk.strip().splitlines()[0] if chunk.strip() else '<empty>'}`\n"
        for i, chunk in enumerate(chunks, 1)
    )
    return f"{len(chunks)}\n\n{chunk_list}\n"


def generate_code_review(chunks: List[str]) -> str:
    """Generate a structured markdown review for the given code chunks."""
    if not chunks:
        return "No code chunks provided -- nothing to review."

    code_context = "\n\n---\n\n".join(
        f"#### Chunk {i + 1}\n```python\n{chunk.strip()}\n```"
        for i, chunk in enumerate(chunks)
    )
    prompt = _REVIEW_PROMPT.format(code_context=code_context)

    client = _get_groq_client()
    if client is not None:
        logger.info("Calling Groq API for code review (%d chunks).", len(chunks))
        try:
            review = _call_groq(client, prompt)
            logger.info("Groq review generated successfully.")
            return review
        except Exception as exc:
            logger.error("Groq API call failed: %s", exc)
            return (
                "## AI Code Review\n\n"
                f"> **Warning:** The AI service returned an error: `{exc}`\n>\n"
                "> Showing placeholder review instead.\n\n"
                + _placeholder_review(chunks)
            )

    logger.info("Groq unavailable -- returning placeholder review.")
    return _placeholder_review(chunks)
