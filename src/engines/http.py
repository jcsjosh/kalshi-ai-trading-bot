"""Throttled, retrying, optionally disk-cached HTTP for public data sources.

The engines read only public endpoints (Kalshi market data, NOAA via IEM, NWS,
Open-Meteo), several of which rate-limit aggressively. Everything goes through
one ``Fetcher`` so that:

* each host gets a minimum spacing between requests,
* 429 / 5xx / connection errors back off exponentially and retry,
* archival responses (settled markets, past forecasts) are cached on disk so a
  backtest never pays for the same request twice.

Failures raise ``FetchError`` with a readable reason; callers decide whether a
missing source is fatal or just means "no opinion on this market".
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse

import httpx

DEFAULT_CACHE_DIR = Path("data/cache/http")
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
USER_AGENT = "kalshi-ai-trading-bot/edge-engines (github.com/ryanfrigo/kalshi-ai-trading-bot)"

# Seconds between requests to the same host. Kalshi's public tier allows ~20/s
# but shares a budget with your trading; NWS and IEM ask for politeness.
HOST_SPACING = {
    "api.elections.kalshi.com": 0.12,
    "demo-api.kalshi.co": 0.12,
    "mesonet.agron.iastate.edu": 0.25,
    "api.weather.gov": 0.25,
}
DEFAULT_SPACING = 0.1


class FetchError(RuntimeError):
    """A request that could not be completed (after retries)."""


class Fetcher:
    def __init__(
        self,
        cache_dir: Optional[Path | str] = DEFAULT_CACHE_DIR,
        timeout: float = 30.0,
        max_retries: int = 5,
        rate_limit_retries: int = 8,
        client: Optional[httpx.Client] = None,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries
        self.rate_limit_retries = rate_limit_retries
        self.client = client or httpx.Client(
            timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True
        )
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "Fetcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- cache ---------------------------------------------------------------

    def _cache_path(self, url: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(url.encode()).hexdigest()[:32]
        host = urlparse(url).netloc.replace(":", "_")
        return self.cache_dir / host / f"{digest}.json"

    def _cache_read(self, url: str, ttl: Optional[float]) -> Optional[Any]:
        path = self._cache_path(url)
        if path is None or not path.exists():
            return None
        if ttl is not None and time.time() - path.stat().st_mtime > ttl:
            return None
        try:
            return json.loads(path.read_text())["body"]
        except (OSError, ValueError, KeyError):
            return None

    def _cache_write(self, url: str, body: Any) -> None:
        path = self._cache_path(url)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({"url": url, "body": body}))
        tmp.replace(path)

    # -- requests ------------------------------------------------------------

    def _throttle(self, host: str) -> None:
        spacing = HOST_SPACING.get(host, DEFAULT_SPACING)
        with self._lock:
            wait = self._last.get(host, 0.0) + spacing - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        as_text: bool = False,
        cache_ttl: Optional[float] = None,
        cache: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """GET ``url`` and return parsed JSON (or text with ``as_text``).

        ``cache=True`` stores the response on disk; ``cache_ttl`` (seconds) bounds
        how stale a cached copy may be (``None`` = forever, right for archives).
        """
        full = url + ("?" + urlencode(params, doseq=True) if params else "")
        if cache:
            hit = self._cache_read(full, cache_ttl)
            if hit is not None:
                return hit

        host = urlparse(full).netloc
        last_err = ""
        attempt = limited = 0
        while attempt < self.max_retries:
            self._throttle(host)
            try:
                resp = self.client.get(full, headers=headers)
            except httpx.HTTPError as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2**attempt, 20))
                attempt += 1
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}"
                if "try again tomorrow" in resp.text.lower():
                    # A daily quota (Open-Meteo free tier) will not clear by retrying.
                    raise FetchError(f"{host}: daily request quota exhausted")
                retry_after = resp.headers.get("retry-after")
                if resp.status_code == 429 and limited < self.rate_limit_retries:
                    # Per-minute limits clear if we wait them out; slow this host down too.
                    limited += 1
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**limited
                    with self._lock:
                        self._last[host] = time.monotonic() + min(delay, 60)
                    time.sleep(min(delay, 60))
                    continue
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                time.sleep(min(delay, 30))
                attempt += 1
                continue
            if resp.status_code >= 400:
                raise FetchError(f"{full}: HTTP {resp.status_code} {resp.text[:200]}")
            body: Any = resp.text if as_text else resp.json()
            if isinstance(body, dict) and body.get("error") is True:
                # Open-Meteo reports quota and validation errors in-band.
                raise FetchError(f"{host}: {body.get('reason', 'error')}")
            if cache:
                self._cache_write(full, body)
            return body
        raise FetchError(f"{full}: gave up after {attempt + limited} attempts ({last_err})")
