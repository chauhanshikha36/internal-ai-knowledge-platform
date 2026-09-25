from pathlib import Path

from app.services.chunking import chunk_code, chunk_markdown, recursive_split

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "Source_Code_Sample.py"


def test_recursive_split_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    parts = recursive_split(text, size=200, overlap=40)
    assert len(parts) > 5
    assert all(len(p) <= 260 for p in parts)
    # overlap: the start of each chunk appears at the end of the previous one
    assert parts[1].split()[0] in parts[0]


def test_python_chunks_are_symbol_aligned():
    chunks = chunk_code(SAMPLE.read_text(), ".py", 1800)
    symbols = {c.metadata["symbol"] for c in chunks}
    assert "DecayProxyRotator.get_proxy" in symbols
    assert "DecayProxyRotator.report_failure" in symbols
    assert "UAFreshnessRotator.get_ua" in symbols
    assert "UAFreshnessRotator.report_block" in symbols
    assert "DecayProxyRotator" in symbols  # class overview
    assert "__main__" in symbols
    assert all(len(c.content) <= 1800 * 1.25 for c in chunks)
    get_proxy = next(c for c in chunks if c.metadata["symbol"] == "DecayProxyRotator.get_proxy")
    assert get_proxy.metadata["start_line"] == 36
    assert "random.choices" in get_proxy.content


def test_python_syntax_error_falls_back_to_generic():
    chunks = chunk_code("def broken(:\n    pass\n" * 50, ".py", 200)
    assert chunks and "symbol" not in chunks[0].metadata


def test_markdown_keeps_heading_path():
    md = "# Guide\nintro\n## Setup\nrun it\n```\n# not a heading\n```\n## Usage\nuse it\n"
    chunks = chunk_markdown(md, 500, 50)
    assert [c.metadata["heading"] for c in chunks] == ["Guide", "Guide > Setup", "Guide > Usage"]
    assert "# not a heading" in chunks[1].content
