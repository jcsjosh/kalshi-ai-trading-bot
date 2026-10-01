"""Predictive distribution over the integer temperature the CLI will report.

    T_cont ~ location-scale Student-t(mu, sigma, df)
    mu     = txn + bias
    sigma  = sqrt((scale * xnd)^2 + floor^2)

The CLI reports whole degrees, so ``P(T = v) = F(v + 0.5) - F(v - 0.5)``.
``bias``, ``scale`` and ``floor`` are fit per station and kind (high / low) on
past NBM forecasts against past CLI reports (see ``calibration.py``); the
Student-t tails absorb the busted forecasts a normal would call impossible.

Observed extremes truncate the support: the high can't finish below the
warmest reading so far (minus one degree of rounding slack), the low can't
finish above the coldest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from scipy import stats

from src.engines.market import Quote

ROUNDING_SLACK = 1  # degrees; observation and CLI rounding can disagree by one
SLACK_WEIGHT = 0.05  # how much of its forecast mass a just-ruled-out reading keeps


@dataclass(frozen=True)
class CalibParams:
    bias: float = 0.0
    scale: float = 1.0
    floor: float = 1.5
    df: float = 6.0

    def sigma(self, xnd: float) -> float:
        return math.sqrt((self.scale * max(xnd, 0.0)) ** 2 + self.floor**2)


@dataclass
class TempDistribution:
    mu: float
    sigma: float
    df: float = 6.0
    lower: Optional[int] = None  # inclusive bound from observations (highs)
    upper: Optional[int] = None  # inclusive bound from observations (lows)

    @classmethod
    def from_forecast(cls, txn: float, xnd: float, params: CalibParams, **bounds) -> "TempDistribution":
        return cls(mu=txn + params.bias, sigma=params.sigma(xnd), df=params.df, **bounds)

    def _cdf(self, x: float) -> float:
        z = (x - self.mu) / self.sigma
        return float(stats.t.cdf(z, self.df)) if self.df and self.df < 200 else float(stats.norm.cdf(z))

    def pmf(self) -> Dict[int, float]:
        """Probability of each whole-degree reading (normalized after truncation)."""
        width = int(math.ceil(10 * self.sigma)) + 3
        lo, hi = int(math.floor(self.mu)) - width, int(math.ceil(self.mu)) + width
        if self.lower is not None:
            lo = max(lo, self.lower - ROUNDING_SLACK)
            hi = max(hi, lo)
        if self.upper is not None:
            hi = min(hi, self.upper + ROUNDING_SLACK)
            lo = min(lo, hi)
        raw = {v: max(self._cdf(v + 0.5) - self._cdf(v - 0.5), 0.0) for v in range(lo, hi + 1)}
        # A reading one degree past an observed extreme is only possible through
        # observation-precision quirks: keep a sliver of it, not its full mass.
        if self.lower is not None and self.lower - ROUNDING_SLACK in raw:
            raw[self.lower - ROUNDING_SLACK] *= SLACK_WEIGHT
        if self.upper is not None and self.upper + ROUNDING_SLACK in raw:
            raw[self.upper + ROUNDING_SLACK] *= SLACK_WEIGHT
        # Tail mass beyond the window folds into the end bins so nothing is lost.
        if self.lower is None:
            raw[lo] += self._cdf(lo - 0.5)
        if self.upper is None:
            raw[hi] += 1.0 - self._cdf(hi + 0.5)
        total = sum(raw.values())
        if total <= 0:
            # Bounds far outside the forecast: everything sits on the bound.
            v = self.lower if self.lower is not None else self.upper
            return {int(v): 1.0}
        return {v: p / total for v, p in raw.items() if p > 0}

    def prob(self, quote: Quote) -> Optional[float]:
        """P(YES) for a temperature market, or None if its strikes aren't numeric."""
        if quote.contains(round(self.mu)) is None:
            return None
        total = 0.0
        for v, p in self.pmf().items():
            hit = quote.contains(v)
            if hit is None:
                return None
            if hit:
                total += p
        return min(max(total, 0.0), 1.0)


def log_score(dist: TempDistribution, actual: int) -> float:
    p = dist.pmf().get(int(actual), 0.0)
    return math.log(max(p, 1e-9))
