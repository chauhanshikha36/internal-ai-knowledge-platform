"""Chunking strategies.

- Prose / PDF / text : recursive splitting (paragraph > line > sentence > word) with overlap
- Markdown           : split on headings first, keep the heading path as metadata
- Python code        : AST based, one chunk per method / function, plus a class overview
- Other code         : line-aware recursive splitting with start/end line metadata
"""
from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field


@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)


TEXT_SEPS = ["\n\n", "\n", ". ", " "]
CODE_SEPS = ["\n\n", "\n"]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    """Split into pieces <= size, keeping separators so joining is lossless."""
    if len(text) <= size:
        return [text]
    sep = next((s for s in seps if s in text), None)
    if sep is None:
        return [text[i : i + size] for i in range(0, len(text), size)]
    rest = seps[seps.index(sep) + 1 :]
    parts = text.split(sep)
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
    out: list[str] = []
    for p in pieces:
        if len(p) > size:
            out.extend(_split(p, size, rest))
        elif p:
            out.append(p)
    return out


def recursive_split(text: str, size: int, overlap: int, seps: list[str] | None = None) -> list[str]:
    seps = seps or TEXT_SEPS
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    cur = ""
    for piece in _split(text, size, seps):
        if cur and len(cur) + len(piece) > size:
            chunks.append(cur.strip())
            tail = cur[-overlap:] if overlap else ""
            # start the overlap on a word boundary
            cur = tail.split(" ", 1)[1] if " " in tail else ""
        cur += piece
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def add_line_numbers(source: str, chunks: list[Chunk]) -> list[Chunk]:
    """Attach start_line/end_line by locating each chunk in the source."""
    cursor = 0
    for c in chunks:
        idx = source.find(c.content, cursor)
        if idx < 0:
            idx = source.find(c.content)
        if idx >= 0:
            start = source.count("\n", 0, idx) + 1
            c.metadata["start_line"] = start
            c.metadata["end_line"] = start + c.content.count("\n")
            cursor = idx + 1
    return chunks


def chunk_prose(text: str, size: int, overlap: int) -> list[Chunk]:
    return add_line_numbers(text, [Chunk(p) for p in recursive_split(text, size, overlap)])


def chunk_markdown(text: str, size: int, overlap: int) -> list[Chunk]:
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_code = False

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(h for _, h in stack), body))
        buf.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
        buf.append(line)
    flush()

    chunks: list[Chunk] = []
    for heading, body in sections:
        for part in recursive_split(body, size, overlap):
            chunks.append(Chunk(part, {"heading": heading} if heading else {}))
    return chunks


def chunk_python(source: str, max_chars: int) -> list[Chunk]:
    """AST based chunking. Raises SyntaxError for invalid code (caller falls back)."""
    tree = ast.parse(source)
    lines = source.splitlines()
    chunks: list[Chunk] = []

    def seg(node) -> tuple[int, int]:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        while start > 1 and lines[start - 2].lstrip().startswith("#"):  # keep leading comments
            start -= 1
        return start, node.end_lineno

    def emit(start: int, end: int, **meta):
        text = textwrap.dedent("\n".join(lines[start - 1 : end])).strip("\n")
        parts = recursive_split(text, max_chars, 0, CODE_SEPS) if len(text) > max_chars else [text]
        if len(parts) > 1 and len(parts[-1]) < 0.15 * max_chars:  # avoid orphan tail chunks
            parts[-2:] = [parts[-2] + "\n" + parts[-1]]
        cursor = 0
        for i, part in enumerate(parts):
            idx = max(text.find(part, cursor), 0)
            cursor = idx + len(part)
            s = start + text.count("\n", 0, idx)
            m = dict(meta, language="python", start_line=s, end_line=s + part.count("\n"))
            if len(parts) > 1:
                m["part"] = f"{i + 1}/{len(parts)}"
            chunks.append(Chunk(part, m))

    pending: list[ast.stmt] = []

    def flush():
        if not pending:
            return
        is_main = any(isinstance(n, ast.If) and "__name__" in ast.unparse(n.test) for n in pending)
        emit(seg(pending[0])[0], pending[-1].end_lineno,
             symbol="__main__" if is_main else "<module>", kind="module")
        pending.clear()

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            flush()
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            overview = [f"class {node.name}", "Methods: " + ", ".join(m.name for m in methods)]
            doc = ast.get_docstring(node)
            if doc:
                overview.append(doc)
            for item in node.body:  # class level attributes
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not (
                    isinstance(item, ast.Expr) and isinstance(getattr(item, "value", None), ast.Constant)
                ):
                    overview.append(textwrap.dedent("\n".join(lines[item.lineno - 1 : item.end_lineno])))
            chunks.append(Chunk("\n".join(overview), {
                "symbol": node.name, "kind": "class_overview", "language": "python",
                "start_line": node.lineno, "end_line": node.end_lineno,
            }))
            for m in methods:
                s, e = seg(m)
                emit(s, e, symbol=f"{node.name}.{m.name}", kind="method", class_name=node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            flush()
            s, e = seg(node)
            emit(s, e, symbol=node.name, kind="function")
        else:
            pending.append(node)
    flush()
    return chunks


def chunk_code(source: str, ext: str, max_chars: int) -> list[Chunk]:
    if ext == ".py":
        try:
            return chunk_python(source, max_chars)
        except SyntaxError:
            pass
    chunks = add_line_numbers(source, [Chunk(p) for p in recursive_split(source, max_chars, 200, CODE_SEPS)])
    for c in chunks:
        c.metadata["language"] = ext.lstrip(".")
    return chunks
