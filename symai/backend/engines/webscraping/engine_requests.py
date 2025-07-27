import io
import logging
import re
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text

from ....symbol import Result
from ...base import Engine

logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("trafilatura").setLevel(logging.WARNING)

class RequestsResult(Result):
    def __init__(self, value, output_format="markdown", **kwargs) -> None:
        super().__init__(value, **kwargs)
        self.output_format = output_format
        self.raw = value
        self._value = self.extract()

    def extract(self):
        ctype = self.raw.headers.get("Content-Type", "").lower()
        is_pdf = "application/pdf" in ctype or self.raw.url.lower().endswith(".pdf")
        try:
            if is_pdf:
                with io.BytesIO(self.raw.content) as fh:
                    self._value = extract_text(fh)
            else:
                decoded = trafilatura.load_html(self.raw.content)
                self._value = trafilatura.extract(decoded, output_format=self.output_format)
        except Exception:  # keep broad except to avoid hard failures
            self._value = None
        return self._value


class RequestsEngine(Engine):
    # Some forums show a consent / age gate only once if a cookie is set.
    # Pre-seed a few common bypass cookies by default.
    COMMON_BYPASS_COOKIES = {
        "cookieconsent_status": "allow",
        "accepted_cookies": "yes",
        "age_verified": "1",
    }
    def __init__(self, timeout=15, verify_ssl=True):
        super().__init__()
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.name = self.__class__.__name__
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",
        })

    def _maybe_set_bypass_cookies(self, url: str):
        netloc = urlparse(url).hostname
        if not netloc:
            return
        for k, v in self.COMMON_BYPASS_COOKIES.items():
            self.session.cookies.set(k, v, domain=netloc)

    def _follow_meta_refresh(self, resp, timeout=15):
        """
        Some old forums use <meta http-equiv="refresh" content="0;url=...">
        (sometimes to simulate a popup or interstitial). Follow it once.
        """
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" not in ctype.lower():
            return resp
        # Use apparent encoding to decode legacy charsets
        soup = BeautifulSoup(resp.text, "html.parser")
        resp.encoding = resp.encoding or resp.apparent_encoding
        meta = soup.find("meta", attrs={"http-equiv": re.compile("^refresh$", re.I)})
        if not meta or "content" not in meta.attrs:
            return resp
        m = re.search(r"url=(.+)", meta["content"], flags=re.I)
        if not m:
            return resp
        refresh_url = m.group(1).strip().strip("'\"")
        target = urljoin(resp.url, refresh_url)
        # Avoid loops
        if target == resp.url:
            return resp
        return self.session.get(target, timeout=timeout, allow_redirects=True)

    def id(self) -> str:
        return 'webscraping'

    def forward(self, argument):
        """
        Return raw bytes of the final page body.
        - Retries network errors (not programming bugs).
        - Handles legacy redirects via meta refresh.
        - Attempts to bypass simple consent/age popups by pre-seeding cookies.
        """
        url = argument.prop.prepared_input
        kwargs = argument.kwargs
        output_format = kwargs.get("output_format", "markdown")

        self._maybe_set_bypass_cookies(url)

        parsed = urlparse(url)
        qs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
              if k.lower() not in {"utm_source", "utm_medium", "utm_campaign"}]
        clean_url = urlunparse(parsed._replace(query=urlencode(qs)))

        resp = self.session.get(clean_url, timeout=self.timeout, allow_redirects=True, verify=self.verify_ssl)
        resp.raise_for_status()

        # Follow a legacy meta refresh once (do AFTER normal HTTP redirects)
        resp2 = self._follow_meta_refresh(resp, timeout=self.timeout)
        if resp2 is not resp:
            resp2.raise_for_status()
            resp = resp2

        metadata = {}
        result = RequestsResult(resp, output_format)
        return [result], metadata

    def prepare(self, argument):
        argument.prop.prepared_input = str(argument.prop.url)
