"""
correction_service.py — AI-powered code correction via Groq.

Receives only the retrieved (relevant) code sections rather than the full file,
keeping the correction focused on problematic chunks and reducing token usage.
Falls back to returning the sections unchanged if Groq is unavailable.
"""
from __future__ import annotations

import logging
from typing import List

from app.services.review_service import _call_groq, _get_groq_client

logger = logging.getLogger(__name__)

_CORRECT_PROMPT = (
    "You are an expert Python developer. Fix ALL bugs, errors, and code quality "
    "issues in the Python code sections below.\n\n"
    "Rules you MUST follow:\n"
    "1. Return ONLY the corrected Python source code — no explanations, no markdown "
    "fences, no prose.\n"
    "2. Fix every bug, logic error, unhandled edge case, PEP 8 violation, and "
    "naming issue you find.\n"
    "3. Do NOT add new features; only correct existing code.\n"
    "4. Preserve the original function/class names unless the name itself is the bug.\n"
    "5. Keep all section header comments (# Function: ...) exactly as-is.\n\n"
    "--- CODE SECTIONS ---\n"
    "{code_sections}\n"
    "---------------------\n\n"
    "Output the corrected sections only."
)


def generate_corrected_code(relevant_sources: List[str]) -> str:
    """Return corrected versions of *relevant_sources* via Groq.

    Only the retrieved chunks (not the full file) are sent, keeping the
    correction focused and token-efficient.  Returns the sections joined
    unchanged if Groq is unavailable or the call fails.
    """
    if not relevant_sources:
        return ""

    code_sections = "\n\n".join(relevant_sources)

    client = _get_groq_client()
    if client is not None:
        prompt = _CORRECT_PROMPT.format(code_sections=code_sections)
        logger.info(
            "Calling Groq to correct %d retrieved section(s).", len(relevant_sources)
        )
        try:
            corrected = _call_groq(client, prompt).strip()
            # Strip markdown fences the model may add despite instructions
            if corrected.startswith("```"):
                lines = corrected.splitlines()
                lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                corrected = "\n".join(lines)
            logger.info("Corrected code generated successfully.")
            return corrected
        except Exception as exc:
            logger.error("Groq correction call failed: %s", exc)
            return code_sections

    logger.info("Groq unavailable — returning sections unchanged.")
    return code_sections

    logger.info("Groq unavailable -- returning original code unchanged.")
    return original_code
