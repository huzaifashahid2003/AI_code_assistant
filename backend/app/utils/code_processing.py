"""
code_processing.py — AST-based code parsing and chunking utilities.

Splits a Python source file into top-level function and class chunks so that
each chunk can be independently embedded and stored in the vector store.
"""
from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class CodeChunk:
    """A single logical unit of Python source code (function or class)."""

    name: str
    chunk_type: str       # "function" | "class"
    source: str
    start_line: int       # 1-based, relative to original file
    end_line: int         # 1-based, relative to original file
    docstring: Optional[str] = field(default=None)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "chunk_type": self.chunk_type,
            "source": self.source,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "docstring": self.docstring,
        }


def chunk_code(code_text: str) -> List[CodeChunk]:
    """Split Python source into top-level function/class chunks.

    Nested definitions are kept inside their parent chunk to ensure each
    chunk is self-contained.  Returns an empty list if the source cannot
    be parsed or contains no top-level definitions.
    """
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return []

    source_lines = code_text.splitlines(keepends=True)
    chunks: List[CodeChunk] = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
        start: int = node.lineno
        end: int = node.end_lineno  # type: ignore[assignment]  # Python 3.8+

        raw_source = "".join(source_lines[start - 1 : end])
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


def chunk_file(file_path: str | Path) -> List[CodeChunk]:
    """Read a .py file from disk and return its code chunks.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError:        If the file is not a .py file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix != ".py":
        raise ValueError(f"Expected a .py file, got: {path.suffix}")
    return chunk_code(path.read_text(encoding="utf-8"))
