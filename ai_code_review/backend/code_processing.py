"""
code_processing.py — Code parsing and chunking utilities.

Responsibilities:
    - Parse uploaded Python source files using the built-in `ast` module.
    - Split source code into logical chunks (functions / classes) so that each
      chunk can be embedded and stored in a vector database independently.

Ready to integrate with an embedding pipeline (see rag_search.py).
"""

import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """
    Represents a single logical unit of Python source code.

    Attributes:
        name:       Qualified name of the function or class (e.g. "MyClass.method").
        chunk_type: Either "function" or "class".
        source:     The raw source text of the chunk.
        start_line: 1-based line number where the chunk starts in the original file.
        end_line:   1-based line number where the chunk ends in the original file.
        docstring:  The first docstring of the chunk, if present.
    """

    name: str
    chunk_type: str          # "function" | "class"
    source: str
    start_line: int
    end_line: int
    docstring: Optional[str] = field(default=None)

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary (useful for JSON responses)."""
        return {
            "name": self.name,
            "chunk_type": self.chunk_type,
            "source": self.source,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "docstring": self.docstring,
        }


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def chunk_code(code_text: str) -> List[CodeChunk]:
    """
    Split a Python source string into top-level function and class chunks.

    Each function definition (``def``) and class definition (``class``) at the
    *module level* produces one :class:`CodeChunk`.  Nested functions/classes
    are intentionally included inside their parent chunk rather than extracted
    separately, keeping each chunk self-contained.

    Args:
        code_text: Raw Python source code as a string.

    Returns:
        A list of :class:`CodeChunk` objects in source order.  Returns an
        empty list if the source contains no top-level definitions or cannot
        be parsed.
    """
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        # Return empty list; caller can decide how to handle unparseable code
        return []

    source_lines = code_text.splitlines(keepends=True)
    chunks: List[CodeChunk] = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
        start = node.lineno          # 1-based
        end = node.end_lineno        # 1-based (Python 3.8+)

        raw_source = "".join(source_lines[start - 1 : end])

        # Dedent so leading indentation at module level does not confuse
        # downstream embedding models.
        dedented_source = textwrap.dedent(raw_source)

        docstring: Optional[str] = ast.get_docstring(node)

        chunks.append(
            CodeChunk(
                name=node.name,
                chunk_type=chunk_type,
                source=dedented_source,
                start_line=start,
                end_line=end,
                docstring=docstring,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# File-level helper
# ---------------------------------------------------------------------------

def chunk_file(file_path: str | Path) -> List[CodeChunk]:
    """
    Read a ``.py`` file from disk and return its code chunks.

    Args:
        file_path: Absolute or relative path to a Python source file.

    Returns:
        List of :class:`CodeChunk` objects (may be empty).

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError:        If the file is not a ``.py`` file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix != ".py":
        raise ValueError(f"Expected a .py file, got: {path.suffix}")

    code_text = path.read_text(encoding="utf-8")
    return chunk_code(code_text)
