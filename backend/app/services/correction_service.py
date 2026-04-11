"""
correction_service.py — AI-powered code correction via Groq.

Calls Groq to return a corrected version of the submitted source code.
Falls back to returning the original unchanged if Groq is unavailable.
"""
from __future__ import annotations

import logging

from app.services.review_service import _call_groq, _get_groq_client

logger = logging.getLogger(__name__)

_CORRECT_PROMPT = (
    "You are an expert Python developer. Your task is to fix ALL bugs, errors, and "
    "code quality issues in the Python code below.\n\n"
    "Rules you MUST follow:\n"
    "1. Return ONLY the corrected Python source code -- no explanations, no markdown "
    "fences, no prose.\n"
    "2. Fix every bug, logic error, unhandled edge case, PEP 8 violation, and "
    "naming issue you find.\n"
    "3. Do NOT add new features; only correct existing code.\n"
    "4. Preserve the original structure, function names, and variable names unless "
    "a name itself is the bug.\n\n"
    "--- ORIGINAL CODE ---\n"
    "{original_code}\n"
    "---------------------\n\n"
    "Output the corrected Python code only."
)


def generate_corrected_code(original_code: str) -> str:
    """Return a corrected version of *original_code* via Groq.

    Returns the original unchanged if Groq is unavailable or the call fails.
    """
    if not original_code.strip():
        return original_code

    client = _get_groq_client()
    if client is not None:
        prompt = _CORRECT_PROMPT.format(original_code=original_code)
        logger.info("Calling Groq API to generate corrected code.")
        try:
            corrected = _call_groq(client, prompt)
            corrected = corrected.strip()
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
            return original_code

    logger.info("Groq unavailable -- returning original code unchanged.")
    return original_code
