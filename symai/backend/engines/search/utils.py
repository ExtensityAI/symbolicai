"""Shared helpers for the search engines.

Engines that ground answers in web sources (OpenAI, Gemini) produce a marker
annotated answer string plus an ordered list of citations. The citation building
pipeline and the URL normalization it depends on are identical across those
engines, so they live here. Engines that return scraped results (Firecrawl,
Parallel) only share the ``Citation`` dataclass, ``normalize_url`` and the
``CitationResultMixin`` repr surface.
"""

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import tldextract


@dataclass
class Citation:
    id: int
    title: str
    url: str
    start: int
    end: int

    def __hash__(self):
        return hash((self.url,))


# Pre-compiled patterns used by strip_markdown_links().
# NOTE: a parenthesized markdown link such as "([label](http://x))"
_RE_WRAPPED_LINK = re.compile(r"\(\s*\[[^\]]+\]\(https?://[^)]+\)\s*\)")
# NOTE: a bare markdown link such as "[label](http://x)"
_RE_BARE_LINK = re.compile(r"\[[^\]]+\]\(https?://[^)]+\)")
# NOTE: parentheses left empty after link removal, e.g. "()"
_RE_EMPTY_PARENS = re.compile(r"\(\s*\)")
# NOTE: comma-only leftovers such as "(, , )"
_RE_COMMA_PARENS = re.compile(r"\(\s*(,\s*)+\)")
# NOTE: runs of two or more whitespace characters
_RE_MULTI_SPACE = re.compile(r"\s{2,}")


def normalize_url(url: str) -> str:
    """Return a canonical URL suitable for citation deduplication.

    Lower-cases scheme/host, strips the trailing path slash (root becomes "/"),
    drops tracking query params (utm_*) and the fragment.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    filtered = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_")
    ]
    query = urlencode(filtered, doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def normalize_domains(
    domains: list[str] | None,
    max_count: int,
) -> list[str]:
    """Normalize a list of raw domain strings to deduped apex hosts, capped.

    Each entry is reduced to its fqdn (apex, subdomains like ``www`` dropped) via
    ``tldextract``; junk inputs (no registrable TLD) yield an empty fqdn and are
    dropped. Duplicates are removed and the result is capped at ``max_count``.
    """
    if not isinstance(domains, list):
        return []
    seen = set()
    out = []
    for domain in domains:
        host = tldextract.extract(domain).fqdn
        if not host or host in seen:
            continue
        seen.add(host)
        out.append(host)
        if len(out) >= max_count:
            break
    return out


def make_title_map(annotations: list[dict] | None) -> dict[str, str]:
    """Map each normalized URL to its first non-empty title from ``annotations``."""
    m = {}
    for a in annotations or []:
        url = a.get("url")
        if not url:
            continue
        nu = normalize_url(url)
        title = (a.get("title") or "").strip()
        if nu not in m and title:
            m[nu] = title
    return m


def strip_markdown_links(text: str) -> str:
    """Remove markdown link syntax (and the resulting empty parens) from ``text``."""
    text = _RE_WRAPPED_LINK.sub("", text)
    text = _RE_BARE_LINK.sub("", text)
    text = _RE_EMPTY_PARENS.sub("", text)
    text = _RE_COMMA_PARENS.sub("", text)
    return _RE_MULTI_SPACE.sub(" ", text).strip()


def snap_end_to_word_boundary(text: str, end: int) -> int:
    # NOTE: Gemini's annotation end_index overshoots by grabbing the first 1-2 chars of the word
    # that follows the cited text (e.g. "...This h" + [id] + "as triggered"). When `end` lands
    # inside an alpha run, move it to a clean boundary so the marker never splits a word:
    # prefer moving LEFT to the run's start when a non-alpha separator precedes it (marker then
    # sits before the whole following word), otherwise move RIGHT to the run's end (marker after
    # the whole word, used when the run has no separator before it, e.g. text start or glued runs).
    # Idempotent: a snapped position sits on a word boundary, where at least one neighbor is
    # non-alpha, so re-applying is a no-op. No-op for OpenAI, whose span ends already sit on
    # word boundaries.
    if not (0 < end < len(text) and text[end - 1].isalpha() and text[end].isalpha()):
        return end
    run_start = end
    while run_start > 0 and text[run_start - 1].isalpha():
        run_start -= 1
    if run_start > 0 and not text[run_start - 1].isalpha():
        return run_start  # marker before the whole following word
    run_end = end
    while run_end < len(text) and text[run_end].isalpha():
        run_end += 1
    return run_end  # no separator before the run -> marker after the whole word


def insert_citation_markers(
    text: str, annotations: list[dict] | None
) -> tuple[str, list[Citation]]:
    """Insert ``[id] (title)`` markers into ``text`` based on URL-citation annotations.

    Returns ``(marked_text, citations)`` where ``citations`` is a list of
    ``Citation`` assigned 1-based in encounter order, each carrying the span of
    its first marker. Sources sharing a normalized URL collapse into one id.
    """

    def assign_id(nu: str) -> int:
        if nu not in id_map:
            # NOTE: ids are 1-based, assigned in encounter order; the next id is
            # derived from the number of distinct sources registered so far.
            cid = len(ordered) + 1
            id_map[nu] = cid
            title = title_map.get(nu) or urlsplit(nu).hostname or ""
            ordered.append((cid, title, nu))
        return id_map[nu]

    title_map = make_title_map(annotations)
    id_map = {}
    first_span = {}
    ordered = []

    url_anns = [a for a in annotations or [] if a.get("type") == "url_citation" and a.get("url")]
    url_anns.sort(key=lambda a: int(a.get("start_index", 0)))

    pieces = []
    cursor = 0
    out_len = 0

    for ann in url_anns:
        start = int(ann.get("start_index", 0))
        end = int(ann.get("end_index", 0))
        end = snap_end_to_word_boundary(text, end)
        if end <= cursor:
            continue  # skip overlapping or backwards spans
        url = ann.get("url")
        nu = normalize_url(url)
        cid = assign_id(nu)
        title = title_map.get(nu) or urlsplit(nu).hostname or ""

        prefix = text[cursor:start]
        prefix_clean = strip_markdown_links(prefix)
        pieces.append(prefix_clean)
        out_len += len(prefix_clean)

        span_text = text[start:end]
        span_clean = strip_markdown_links(span_text)
        span_end_out = out_len + len(span_clean)
        pieces.append(span_clean)
        out_len = span_end_out

        marker = f"[{cid}] ({title})\n"
        marker_start_out = out_len
        marker_end_out = out_len + len(marker)
        if cid not in first_span:
            first_span[cid] = (marker_start_out, marker_end_out)
        pieces.append(marker)
        out_len = marker_end_out
        cursor = end

    tail_clean = strip_markdown_links(text[cursor:])
    pieces.append(tail_clean)
    replaced = "".join(pieces)

    starts_ends = {cid: first_span.get(cid, (0, 0)) for cid, _, _ in ordered}
    citations = [
        Citation(id=cid, title=title, url=nu, start=starts_ends[cid][0], end=starts_ends[cid][1])
        for cid, title, nu in ordered
    ]
    return replaced, citations


class CitationResultMixin:
    """Result mixin for a citation-bearing string value.

    The subclass is expected to set ``self._value`` (the marker-annotated text, or
    None) and ``self._citations`` (a list[Citation]) during construction. The
    ``raw`` attribute is provided by Result.
    """

    _value: str | None
    _citations: list[Citation]
    raw: Any

    def get_citations(self) -> list[Citation]:
        return self._citations

    def __str__(self) -> str:
        if isinstance(self._value, str) and self._value:
            return self._value
        try:
            return json.dumps(self.raw, indent=2)
        except TypeError:
            return str(self.raw)

    def _repr_html_(self) -> str:
        if isinstance(self._value, str) and self._value:
            return f"<pre>{self._value}</pre>"
        try:
            return f"<pre>{json.dumps(self.raw, indent=2)}</pre>"
        except Exception:
            return f"<pre>{self.raw!s}</pre>"
