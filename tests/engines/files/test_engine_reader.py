"""Tests for the markitdown-based file reader engine.

Tests are split into three tiers:
  1. Unit tests on the engine internals (extension sets, SupportedFileType enum).
     These don't need symai init.
  2. Integration tests via FileReader / Symbol.open() for plain-text formats.
     These need `import symai` (config + engine registry).
  3. Integration tests for rich formats and URL-based converters.
     These need sample fixtures and (for URLs) network access.
"""

import tempfile
from pathlib import Path

import pytest
from box import Box, BoxList

# ---------------------------------------------------------------------------
# Test data directory (relative to repo root)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# -- plain-text fixtures already in tests/data/ --
PLAIN_TEXT_FILE = DATA_DIR / "sample.txt"

# -- rich-format fixtures the user should add to tests/data/ --
PDF_FILE = DATA_DIR / "sample.pdf"
DOCX_FILE = DATA_DIR / "sample.docx"
PPTX_FILE = DATA_DIR / "sample.pptx"
XLSX_FILE = DATA_DIR / "sample.xlsx"
HTML_FILE = DATA_DIR / "sample.html"
IPYNB_FILE = DATA_DIR / "sample.ipynb"
EPUB_FILE = DATA_DIR / "sample.epub"

# -- image fixtures in tests/data/ --
JPG_FILE = DATA_DIR / "sample.jpg"
PNG_FILE = DATA_DIR / "sample.png"

# -- archive fixture in tests/data/ --
ZIP_FILE = DATA_DIR / "sample.zip"

# -- legacy spreadsheet fixture in tests/data/ --
XLS_FILE = DATA_DIR / "sample.xls"

# -- audio fixture in tests/data/ --
AUDIO_FILE = DATA_DIR / "sample.mp3"


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1: unit tests (no symai init needed)
# ═══════════════════════════════════════════════════════════════════════════

from symai.backend.engines.files.engine_reader import (  # noqa: E402
    _IMAGE_EXTS,
    _PLAIN_TEXT_EXTS,
    _RICH_FORMAT_EXTS,
    SupportedFileType,
)


class TestExtensionSets:
    def test_sets_do_not_overlap(self):
        assert not (_PLAIN_TEXT_EXTS & _RICH_FORMAT_EXTS)
        assert not (_PLAIN_TEXT_EXTS & _IMAGE_EXTS)
        assert not (_RICH_FORMAT_EXTS & _IMAGE_EXTS)

    def test_enum_covers_all_extensions(self):
        all_exts = _PLAIN_TEXT_EXTS | _RICH_FORMAT_EXTS | _IMAGE_EXTS
        for ft in SupportedFileType:
            assert ft.value in all_exts

    def test_common_extensions_classified_correctly(self):
        for ext in (".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".tsv", ".toml", ".xml", ".log"):
            assert ext in _PLAIN_TEXT_EXTS, f"{ext} should be plain text"
        for ext in (".jpg", ".jpeg", ".png"):
            assert ext in _IMAGE_EXTS, f"{ext} should be image"
        for ext in (
            ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".epub", ".ipynb",
            ".mp3", ".wav", ".m4a", ".mp4", ".zip",
        ):
            assert ext in _RICH_FORMAT_EXTS, f"{ext} should be rich format"


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2: integration via FileReader / Symbol.open() — plain-text formats
# ═══════════════════════════════════════════════════════════════════════════

from symai import Symbol  # noqa: E402
from symai.components import FileReader  # noqa: E402


class TestPlainTextViaFileReader:
    """Tests that go through the full engine pipeline for plain-text files."""

    def test_read_txt(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("hello world")
        try:
            result = Symbol(f.name).open()
            assert result.value == "hello world"
        finally:
            Path(f.name).unlink()

    def test_read_md(self):
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("# Heading\n\nParagraph.")
        try:
            result = Symbol(f.name).open()
            assert "# Heading" in result.value
        finally:
            Path(f.name).unlink()

    def test_read_py(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("print('hi')")
        try:
            result = Symbol(f.name).open()
            assert "print" in result.value
        finally:
            Path(f.name).unlink()

    def test_read_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write('{"key": "value"}')
        try:
            result = Symbol(f.name).open()
            assert isinstance(result.value, dict)
            assert result.value["key"] == "value"
        finally:
            Path(f.name).unlink()

    def test_read_yaml(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("key: value\n")
        try:
            result = Symbol(f.name).open()
            assert isinstance(result.value, dict)
            assert result.value["key"] == "value"
        finally:
            Path(f.name).unlink()

    def test_read_csv(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("name,age\nAlice,30\n")
        try:
            result = Symbol(f.name).open()
            assert isinstance(result.value, list)
            assert result.value[0]["name"] == "Alice"
        finally:
            Path(f.name).unlink()

    def test_read_tsv(self):
        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            f.write("col1\tcol2\nval1\tval2\n")
        try:
            result = Symbol(f.name).open()
            assert isinstance(result.value, list)
            assert result.value[0]["col1"] == "val1"
        finally:
            Path(f.name).unlink()

    def test_read_toml(self):
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write('[section]\nkey = "val"\n')
        try:
            result = Symbol(f.name).open()
            assert isinstance(result.value, dict)
            assert result.value["section"]["key"] == "val"
        finally:
            Path(f.name).unlink()

    def test_read_log(self):
        with tempfile.NamedTemporaryFile(suffix=".log", mode="w", delete=False) as f:
            f.write("[INFO] startup complete\n")
        try:
            result = Symbol(f.name).open()
            assert "startup complete" in result.value
        finally:
            Path(f.name).unlink()

    def test_existing_txt_fixture(self):
        if not PLAIN_TEXT_FILE.exists():
            pytest.skip(f"Fixture not found: {PLAIN_TEXT_FILE}")
        result = Symbol(str(PLAIN_TEXT_FILE)).open()
        assert len(result.value) > 100, "sample.txt should be substantial"

    def test_empty_file_returns_empty_string(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            pass  # 0 bytes
        try:
            result = Symbol(f.name).open()
            assert result.value == ""
        finally:
            Path(f.name).unlink()

    def test_file_not_found_raises(self):
        with pytest.raises(AssertionError, match="File does not exist"):
            Symbol("/nonexistent/file.txt").open()

    def test_no_path_raises(self):
        with pytest.raises(ValueError, match="Path is not provided"):
            Symbol().open()

    def test_unknown_extension_auto_reads_as_text(self):
        """auto backend reads unknown extensions as plain text."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
            f.write("mystery format")
        try:
            result = Symbol(f.name).open()
            assert result.value == "mystery format"
        finally:
            Path(f.name).unlink()

    def test_unknown_extension_standard_raises(self):
        """standard backend rejects unknown extensions."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
            f.write("mystery format")
        try:
            with pytest.raises(ValueError, match="Unsupported file extension"):
                Symbol(f.name).open(backend="standard")
        finally:
            Path(f.name).unlink()


class TestFileReaderComponent:
    """Tests through the FileReader Expression component."""

    def test_single_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("reader test")
        try:
            reader = FileReader()
            result = reader(f.name)
            assert isinstance(result.value, list)
            assert result.value[0] == "reader test"
        finally:
            Path(f.name).unlink()

    def test_multiple_files(self):
        paths = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
                f.write(f"content {i}")
            paths.append(f.name)
        try:
            reader = FileReader()
            result = reader(paths)
            assert len(result.value) == 3
            for i in range(3):
                assert f"content {i}" in result.value[i]
        finally:
            for p in paths:
                Path(p).unlink()

    def test_get_files_finds_accepted_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for ext in (
                ".txt", ".md", ".py", ".json", ".csv",
                ".pdf", ".docx", ".html", ".jpg", ".png", ".zip", ".mp3",
            ):
                (Path(tmpdir) / f"test{ext}").write_text("x")
            # Also create an unsupported file
            (Path(tmpdir) / "test.exe").write_text("x")
            files = FileReader.get_files(tmpdir)
            exts_found = {Path(f).suffix for f in files}
            assert ".exe" not in exts_found
            assert ".txt" in exts_found
            assert ".pdf" in exts_found
            assert ".jpg" in exts_found
            assert ".zip" in exts_found
            assert ".mp3" in exts_found

    def test_get_files_respects_max_depth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "top.txt").write_text("x")
            deep = Path(tmpdir) / "a" / "b"
            deep.mkdir(parents=True)
            (deep / "deep.txt").write_text("x")
            shallow = FileReader.get_files(tmpdir, max_depth=0)
            assert len(shallow) == 1
            all_files = FileReader.get_files(tmpdir, max_depth=5)
            assert len(all_files) == 2


class TestAsBox:
    """Tests the as_box flag for structured parsing via python-box."""

    def test_json_as_box(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write('{"name": "Alice", "age": 30}')
        try:
            result = Symbol(f.name).open(as_box=True)
            assert isinstance(result.value, Box)
            assert result.value.name == "Alice"
            assert result.value.age == 30
        finally:
            Path(f.name).unlink()

    def test_yaml_as_box(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("name: Bob\nage: 25\n")
        try:
            result = Symbol(f.name).open(as_box=True)
            assert isinstance(result.value, Box)
            assert result.value.name == "Bob"
            assert result.value.age == 25
        finally:
            Path(f.name).unlink()

    def test_toml_as_box(self):
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write('[server]\nhost = "localhost"\nport = 8080\n')
        try:
            result = Symbol(f.name).open(as_box=True)
            assert isinstance(result.value, Box)
            assert result.value.server.host == "localhost"
            assert result.value.server.port == 8080
        finally:
            Path(f.name).unlink()

    def test_csv_as_boxlist(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("name,age\nAlice,30\nBob,25\n")
        try:
            result = Symbol(f.name).open(as_box=True)
            assert isinstance(result.value, BoxList)
            assert len(result.value) == 2
            assert result.value[0].name == "Alice"
        finally:
            Path(f.name).unlink()

    def test_as_box_unsupported_ext_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("plain text")
        try:
            with pytest.raises(ValueError, match="as_box is not supported"):
                Symbol(f.name).open(as_box=True)
        finally:
            Path(f.name).unlink()


class TestImageReading:
    """Tests that images return RGB numpy arrays via the standard backend."""

    def test_jpg_returns_rgb_array(self):
        if not JPG_FILE.exists():
            pytest.skip(f"Fixture not found: {JPG_FILE}")
        result = Symbol(str(JPG_FILE)).open()
        assert hasattr(result.value, "shape"), "Expected numpy array"
        assert result.value.ndim == 3
        assert result.value.shape[2] == 3, "Expected 3 channels (RGB)"

    def test_png_returns_rgb_array(self):
        if not PNG_FILE.exists():
            pytest.skip(f"Fixture not found: {PNG_FILE}")
        result = Symbol(str(PNG_FILE)).open()
        assert hasattr(result.value, "shape"), "Expected numpy array"
        assert result.value.ndim == 3
        assert result.value.shape[2] == 3, "Expected 3 channels (RGB)"

    def test_image_is_rgb_not_bgr(self):
        """Verify the image is returned as RGB by comparing with direct cv2 read."""
        import cv2
        import numpy as np

        if not JPG_FILE.exists():
            pytest.skip(f"Fixture not found: {JPG_FILE}")
        result = Symbol(str(JPG_FILE)).open()
        bgr = cv2.imread(str(JPG_FILE))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        assert np.array_equal(result.value, rgb)


class TestMarkitdownBackend:
    """Tests the backend='markitdown' kwarg."""

    def test_markitdown_backend_on_plain_text(self):
        """backend='markitdown' on a .txt should still return content."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("markitdown backend test")
        try:
            result = Symbol(f.name).open(backend="markitdown")
            assert "markitdown backend test" in result.value
        finally:
            Path(f.name).unlink()

    def test_standard_backend_rejects_rich_format(self):
        """Standard backend on a rich format should suggest markitdown."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", mode="w", delete=False) as f:
            f.write("fake pdf")
        try:
            with pytest.raises(ValueError, match=r"standard.*markitdown"):
                Symbol(f.name).open(backend="standard")
        finally:
            Path(f.name).unlink()

    def test_markitdown_backend_rejects_unknown_ext(self):
        """Markitdown backend on an unknown extension should list supported formats."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
            f.write("unknown")
        try:
            with pytest.raises(ValueError, match="not supported by the markitdown"):
                Symbol(f.name).open(backend="markitdown")
        finally:
            Path(f.name).unlink()


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3: rich formats via markitdown
# ═══════════════════════════════════════════════════════════════════════════


class TestRichFormats:
    """Tests for formats that require markitdown converters (backend='markitdown')."""

    def test_pdf(self):
        if not PDF_FILE.exists():
            pytest.skip(f"Fixture not found: {PDF_FILE}")
        result = Symbol(str(PDF_FILE)).open(backend="markitdown")
        assert len(result.value) > 100, "PDF should produce substantial text"

    def test_html_inline(self):
        with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False) as f:
            f.write("<html><body><h1>Title</h1><p>Hello world</p></body></html>")
        try:
            result = Symbol(f.name).open(backend="markitdown")
            assert "Title" in result.value
            assert "Hello" in result.value
        finally:
            Path(f.name).unlink()

    def test_csv_via_markitdown(self):
        """CSV routed through markitdown produces a markdown table."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("x,y\n1,2\n3,4\n")
        try:
            result = Symbol(f.name).open(backend="markitdown")
            # markitdown converts CSV to a markdown table with pipes
            assert "|" in result.value or "1" in result.value
        finally:
            Path(f.name).unlink()

    def test_docx(self):
        if not DOCX_FILE.exists():
            pytest.skip(f"Fixture not found: {DOCX_FILE}")
        result = Symbol(str(DOCX_FILE)).open(backend="markitdown")
        assert len(result.value) > 0, "DOCX should produce text"

    def test_pptx(self):
        if not PPTX_FILE.exists():
            pytest.skip(f"Fixture not found: {PPTX_FILE}")
        result = Symbol(str(PPTX_FILE)).open(backend="markitdown")
        assert len(result.value) > 0, "PPTX should produce text"

    def test_xlsx(self):
        if not XLSX_FILE.exists():
            pytest.skip(f"Fixture not found: {XLSX_FILE}")
        result = Symbol(str(XLSX_FILE)).open(backend="markitdown")
        assert len(result.value) > 0, "XLSX should produce text"

    def test_html_fixture(self):
        if not HTML_FILE.exists():
            pytest.skip(f"Fixture not found: {HTML_FILE}")
        result = Symbol(str(HTML_FILE)).open(backend="markitdown")
        assert len(result.value) > 0, "HTML should produce text"

    def test_ipynb(self):
        if not IPYNB_FILE.exists():
            pytest.skip(f"Fixture not found: {IPYNB_FILE}")
        result = Symbol(str(IPYNB_FILE)).open(backend="markitdown")
        assert len(result.value) > 0, "Notebook should produce text"

    def test_epub(self):
        if not EPUB_FILE.exists():
            pytest.skip(f"Fixture not found: {EPUB_FILE}")
        result = Symbol(str(EPUB_FILE)).open(backend="markitdown")
        assert len(result.value) > 100, "EPUB should produce substantial text"

    def test_xls(self):
        if not XLS_FILE.exists():
            pytest.skip(f"Fixture not found: {XLS_FILE}")
        result = Symbol(str(XLS_FILE)).open(backend="markitdown")
        assert len(result.value) > 0, "XLS should produce text"

    def test_zip(self):
        if not ZIP_FILE.exists():
            pytest.skip(f"Fixture not found: {ZIP_FILE}")
        result = Symbol(str(ZIP_FILE)).open(backend="markitdown")
        assert result.value is not None, "ZIP should produce some output"

    def test_jpg(self):
        if not JPG_FILE.exists():
            pytest.skip(f"Fixture not found: {JPG_FILE}")
        result = Symbol(str(JPG_FILE)).open(backend="markitdown")
        assert result.value is not None

    def test_png(self):
        if not PNG_FILE.exists():
            pytest.skip(f"Fixture not found: {PNG_FILE}")
        result = Symbol(str(PNG_FILE)).open(backend="markitdown")
        assert result.value is not None

    def test_audio(self):
        if not AUDIO_FILE.exists():
            pytest.skip(f"Fixture not found: {AUDIO_FILE}")
        result = Symbol(str(AUDIO_FILE)).open(backend="markitdown")
        assert result.value is not None


class TestFileReaderBatch:
    """Batch-read all available sample fixtures via FileReader and verify list output."""

    ALL_FIXTURES = (
        PLAIN_TEXT_FILE, PDF_FILE, DOCX_FILE, PPTX_FILE, XLSX_FILE,
        HTML_FILE, IPYNB_FILE, EPUB_FILE, JPG_FILE, PNG_FILE,
        ZIP_FILE, XLS_FILE, AUDIO_FILE,
    )

    def test_parallel_batch_read_and_dump(self):
        """Parallel batch-read all fixtures via FileReader and dump to /tmp."""
        paths = [str(f) for f in self.ALL_FIXTURES if f.exists()]
        if len(paths) < 3:
            pytest.skip("Need at least 3 sample fixtures for a meaningful batch test")
        reader = FileReader()
        result = reader(paths, workers=4, backend="markitdown")
        assert isinstance(result.value, list)
        assert len(result.value) == len(paths)
        out_dir = Path("/tmp/markitdown_output_filereader")
        out_dir.mkdir(exist_ok=True)
        for i, (path, content) in enumerate(zip(paths, result.value, strict=True)):
            assert isinstance(content, str), f"Item {i} ({path}) should be a string"
            assert len(content) > 0, f"Item {i} ({path}) should be non-empty"
            (out_dir / f"{Path(path).name}.md").write_text(content)
