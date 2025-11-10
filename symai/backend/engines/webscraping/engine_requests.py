"""
WARNING: This module implements a naive web scraping engine meant for light
testing. It does not prevent IP bans, bot detection, or terms-of-service
violations. Use only where scraping is legally permitted and respect each
site's robots directives. For production workloads, add robust rate limiting,
consent handling, rotating proxies/VPNs, and ongoing monitoring to avoid
service disruption.
"""

import io
import logging
import re
from typing import Any, ClassVar
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from requests.structures import CaseInsensitiveDict

from ....symbol import Result
from ....utils import UserMessage
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
    """
    Lightweight HTTP/Playwright fetching pipeline for content extraction.

    The engine favors clarity over stealth. Helper methods normalize cookie
    metadata before handing it to Playwright so that the headless browser and
    the requests session stay aligned.
    """

    COMMON_BYPASS_COOKIES: ClassVar[dict[str, str]] = {
        # Some forums display consent or age gates once if a friendly cookie is set.
        "cookieconsent_status": "allow",
        "accepted_cookies": "yes",
        "age_verified": "1",
    }

    DEFAULT_HEADERS: ClassVar[dict[str, str]] = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "DNT": "1",
    }

    _SAMESITE_CANONICAL: ClassVar[dict[str, str]] = {
        "strict": "Strict",
        "lax": "Lax",
        "none": "None",
    }

    def __init__(self, timeout=15, verify_ssl=True, user_agent=None):
        """
        Args:
            timeout: Seconds to wait for network operations before aborting.
            verify_ssl: Toggle for TLS certificate verification.
            user_agent: Optional override for the default desktop Chrome UA.
        """
        super().__init__()
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.name = self.__class__.__name__

        headers = dict(self.DEFAULT_HEADERS)
        if user_agent:
            headers["User-Agent"] = user_agent

        self.session = requests.Session()
        self.session.headers.update(headers)

    def _maybe_set_bypass_cookies(self, url: str):
        netloc = urlparse(url).hostname
        if not netloc:
            return
        for k, v in self.COMMON_BYPASS_COOKIES.items():
            self.session.cookies.set(k, v, domain=netloc)

    @staticmethod
    def _normalize_http_only(raw_value, key_present):
        """
        Playwright expects a boolean. Cookie metadata can arrive as strings,
        numbers, or placeholder objects, so normalize defensively.
        """
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in {"false", "0", "no"}:
                return False
            if normalized in {"true", "1", "yes"}:
                return True
        if raw_value is None:
            return key_present
        return bool(raw_value)

    @classmethod
    def _normalize_same_site(cls, raw_value):
        if raw_value is None:
            return None
        normalized = str(raw_value).strip().lower()
        return cls._SAMESITE_CANONICAL.get(normalized)

    def _playwright_cookie_payload(self, cookie, hostname):
        """
        Convert a requests cookie into Playwright-friendly format or return None
        if the cookie does not apply to the hostname.
        """
        domain = (cookie.domain or hostname).lstrip(".")
        if not hostname.endswith(domain):
            return None

        rest_attrs = {k.lower(): v for k, v in cookie._rest.items()}
        http_only = self._normalize_http_only(rest_attrs.get("httponly"), "httponly" in rest_attrs)
        payload = {
            "name": cookie.name,
            "value": cookie.value,
            "domain": cookie.domain or hostname,
            "path": cookie.path or "/",
            "httpOnly": http_only,
            "secure": cookie.secure,
        }
        if cookie.expires:
            payload["expires"] = cookie.expires

        same_site = self._normalize_same_site(rest_attrs.get("samesite"))
        if same_site:
            payload["sameSite"] = same_site
        return payload

    def _collect_playwright_cookies(self, hostname: str) -> list[dict[str, Any]]:
        if not hostname:
            return []
        cookie_payload = []
        for cookie in self.session.cookies:
            payload = self._playwright_cookie_payload(cookie, hostname)
            if payload:
                cookie_payload.append(payload)
        return cookie_payload

    @staticmethod
    def _add_cookies_to_context(context, cookie_payload: list[dict[str, Any]]) -> None:
        if cookie_payload:
            context.add_cookies(cookie_payload)

    @staticmethod
    def _navigate_playwright_page(page, url: str, wait_selector: str | None, wait_until: str, timeout_ms: int, timeout_error):
        try:
            response = page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            if wait_selector:
                page.wait_for_selector(wait_selector, timeout=timeout_ms)
            return response, None
        except timeout_error as exc:
            return None, exc

    @staticmethod
    def _safe_page_content(page) -> str:
        try:
            return page.content()
        except Exception:
            return ""

    def _sync_cookies_from_context(self, context) -> None:
        for cookie in context.cookies():
            self.session.cookies.set(
                cookie["name"],
                cookie["value"],
                domain=cookie.get("domain"),
                path=cookie.get("path", "/"),
            )

    @staticmethod
    def _rendered_response_metadata(page, response):
        final_url = page.url
        status = response.status if response is not None else 200
        headers = CaseInsensitiveDict(response.headers if response is not None else {})
        if "content-type" not in headers:
            headers["Content-Type"] = "text/html; charset=utf-8"
        return final_url, status, headers

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

    def _fetch_with_playwright(self, url: str, wait_selector: str | None = None, wait_until: str = "networkidle", timeout: float | None = None):
        """
        Render the target URL in a headless browser to execute JavaScript and
        return a synthetic ``requests.Response`` object to keep downstream
        processing consistent with the non-JS path.
        """
        try:
            # Playwright is optional; import only when JS rendering is requested.
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError # noqa
            from playwright.sync_api import sync_playwright # noqa
            logging.getLogger("playwright").setLevel(logging.WARNING)
        except ImportError as exc:
            msg = "Playwright is not installed. Install symbolicai[webscraping] with Playwright extras to enable render_js."
            UserMessage(msg)
            raise RuntimeError(msg) from exc

        timeout_seconds = timeout if timeout is not None else self.timeout
        timeout_ms = max(int(timeout_seconds * 1000), 0)
        user_agent = self.session.headers.get("User-Agent")

        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        cookie_payload = self._collect_playwright_cookies(hostname)

        content = ""
        final_url = url
        status = 200
        headers = CaseInsensitiveDict()

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=user_agent,
                java_script_enabled=True,
                ignore_https_errors=not self.verify_ssl,
            )
            try:
                self._add_cookies_to_context(context, cookie_payload)
                page = context.new_page()

                response, navigation_error = self._navigate_playwright_page(
                    page,
                    url,
                    wait_selector,
                    wait_until,
                    timeout_ms,
                    PlaywrightTimeoutError,
                )
                content = self._safe_page_content(page)
                self._sync_cookies_from_context(context)

                final_url, status, headers = self._rendered_response_metadata(page, response)
                if navigation_error and not content:
                    msg = f"Playwright timed out while rendering {url}"
                    UserMessage(msg)
                    raise requests.exceptions.Timeout(msg) from navigation_error
            finally:
                context.close()
                browser.close()

        rendered_response = requests.Response()
        rendered_response.status_code = status
        rendered_response._content = content.encode("utf-8", errors="replace")
        rendered_response.url = final_url
        rendered_response.headers = headers
        rendered_response.encoding = "utf-8"
        return rendered_response

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

        render_js = kwargs.get("render_js")
        render_wait_selector = kwargs.get("render_wait_selector")
        render_wait_until = kwargs.get("render_wait_until", "networkidle")
        render_timeout = kwargs.get("render_timeout")

        # Prefer fast requests path unless the caller opts into JS rendering.
        if render_js:
            resp = self._fetch_with_playwright(
                clean_url,
                wait_selector=render_wait_selector,
                wait_until=render_wait_until,
                timeout=render_timeout,
            )
        else:
            resp = self.session.get(clean_url, timeout=self.timeout, allow_redirects=True, verify=self.verify_ssl)
        resp.raise_for_status()

        # Follow a legacy meta refresh once (do AFTER normal HTTP redirects)
        resp2 = self._follow_meta_refresh(resp, timeout=self.timeout)
        if resp2 is not resp:
            resp2.raise_for_status()
            resp = resp2

        metadata = {
            "response_source": "playwright" if render_js else "requests",
            "render_js": bool(render_js),
            "final_url": resp.url,
        }
        result = RequestsResult(resp, output_format)
        return [result], metadata

    def prepare(self, argument):
        argument.prop.prepared_input = str(argument.prop.url)
