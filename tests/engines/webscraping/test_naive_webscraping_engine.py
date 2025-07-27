import json
import logging

import pytest

from symai.backend.engines.webscraping.engine_requests import RequestsResult
from symai.extended import Interface
from symai.utils import semassert

logging.getLogger("trafilatura").setLevel(logging.WARNING)

try:
    import bs4
    import trafilatura
except ImportError:
    raise ImportError("trafilatura and/or bs4 not installed. Please install them.")

scraper = Interface('naive_webscraping')

@pytest.mark.parametrize("output_format", ["txt", "markdown", "csv", "json", "html", "xml"])
def test_naive_webscraping(output_format):
    url = "https://trafilatura.readthedocs.io/en/latest/crawls.html"
    rsp = scraper(url, output_format=output_format)

    assert isinstance(rsp, RequestsResult), f"Expected RequestsResult, got {type(rsp)}"
    semassert(rsp is not None, f"Expected a non-empty response for format '{output_format}'")

    content = str(rsp).strip()

    if output_format == "json":
        try:
            parsed = json.loads(content)
            semassert(isinstance(parsed, dict), "JSON output is not a valid JSON object")
        except json.JSONDecodeError as exc:
            semassert(False, f"Invalid JSON output: {exc}")
    elif output_format in ("html", "xml"):
        semassert(content.startswith("<"), f"Expected {output_format} content to start with '<'")
    else:  # txt, markdown, or csv
        semassert(len(content) > 0, f"{output_format} output is empty")
