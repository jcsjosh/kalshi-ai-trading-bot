"""Kalshi trading fees, priced per series from the exchange's own metadata.

Kalshi's fee schedule (effective 2026-07-07)::

    taker fee = round_up_to_cent(M * 0.07   * C * P * (1 - P))
    maker fee = round_up_to_cent(M * 0.0175 * C * P * (1 - P))   # only where enabled

``P`` is the price in dollars, ``C`` the contract count and ``M`` the series'
``fee_multiplier`` (1, 0.5 on some index series, 0 on fee-free series). Maker
fees apply only to series whose ``fee_type`` includes maker fees; every other
resting order is free. There is no settlement fee.

Every series reports ``fee_type`` and ``fee_multiplier`` via ``GET /series/{t}``,
so engines read the real schedule instead of assuming one. Unknown fee types
are priced with maker fees on: an evaluator that overestimates costs passes on
a marginal trade; one that underestimates them bleeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

TAKER_COEFFICIENT = 0.07
MAKER_COEFFICIENT = 0.0175
_FREE_MAKER_TYPES = {"quadratic"}


def _round_up_cent(dollars: float) -> float:
    # The epsilon keeps exact cents (0.07000000001 from float noise) from rounding up.
    return math.ceil(dollars * 100.0 - 1e-9) / 100.0


@dataclass(frozen=True)
class FeeSchedule:
    fee_type: str = "quadratic"
    multiplier: float = 1.0

    @classmethod
    def from_series(cls, series: Dict[str, Any]) -> "FeeSchedule":
        raw = series.get("series", series)
        mult = raw.get("fee_multiplier")
        return cls(
            fee_type=str(raw.get("fee_type") or "quadratic"),
            multiplier=float(mult) if mult is not None else 1.0,
        )

    @property
    def maker_fees(self) -> bool:
        return self.fee_type not in _FREE_MAKER_TYPES

    def fee(self, count: float, price: float, role: str) -> float:
        """Total fee in dollars for one order of ``count`` contracts at ``price``."""
        if count <= 0 or not 0.0 < price < 1.0:
            return 0.0
        if role == "maker":
            if not self.maker_fees:
                return 0.0
            coef = MAKER_COEFFICIENT
        else:
            coef = TAKER_COEFFICIENT
        return _round_up_cent(self.multiplier * coef * count * price * (1.0 - price))

    def per_contract(self, count: float, price: float, role: str) -> float:
        """Average fee per contract at this order size (rounding makes small orders dearer)."""
        return self.fee(max(count, 1), price, role) / max(count, 1)


DEFAULT_SCHEDULE = FeeSchedule()


class SeriesFees:
    """Lazy per-series fee lookup backed by ``GET /series/{ticker}``."""

    def __init__(self, fetcher=None, base_url: str = "https://api.elections.kalshi.com/trade-api/v2"):
        self.fetcher = fetcher
        self.base_url = base_url
        self._cache: Dict[str, FeeSchedule] = {}

    def seed(self, series_ticker: str, series: Dict[str, Any]) -> None:
        self._cache[series_ticker] = FeeSchedule.from_series(series)

    def get(self, series_ticker: str) -> FeeSchedule:
        if series_ticker in self._cache:
            return self._cache[series_ticker]
        schedule = DEFAULT_SCHEDULE
        if self.fetcher is not None:
            try:
                data = self.fetcher.get(
                    f"{self.base_url}/series/{series_ticker}", cache=True, cache_ttl=86400
                )
                schedule = FeeSchedule.from_series(data)
            except Exception:
                # Pessimistic default: maker fees on.
                schedule = FeeSchedule(fee_type="unknown", multiplier=1.0)
        self._cache[series_ticker] = schedule
        return schedule


def series_of(ticker: Optional[str]) -> str:
    return (ticker or "").split("-", 1)[0]
