"""Fit the weather model's calibration on past NBM forecasts vs past CLI reports.

Per kind (high / low) we fit pooled ``scale``, ``floor`` and ``bias`` by
maximum likelihood of the whole-degree outcome under a Student-t. Station
biases are then estimated from each station's own residuals and shrunk toward
the pooled bias (``n / (n + prior_n)``): a station with a handful of days can't
claim a quirk the data doesn't support, while one with months of evidence (NYC's
Central Park running cool) gets its own correction.

Everything is vectorized so the backtest can refit walk-forward every day.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy import optimize, stats

from src.engines.weather.model import CalibParams

DEFAULT_CALIBRATION_PATH = Path("data/engines/weather_calibration.json")
DF = 6.0
STATION_PRIOR_N = 15.0

# Used until a calibration has been fit: NBM's own spread, widened, no bias.
DEFAULT_PARAMS = {"high": CalibParams(0.0, 1.2, 1.5, DF), "low": CalibParams(0.0, 1.2, 1.5, DF)}


@dataclass(frozen=True)
class Sample:
    station: str
    kind: str
    target: date
    txn: float
    xnd: float
    actual: int


def _nll(theta, txn, xnd, actual, df=DF):
    bias, log_scale, log_floor = theta
    sigma = np.sqrt((np.exp(log_scale) * xnd) ** 2 + np.exp(log_floor) ** 2)
    mu = txn + bias
    p = stats.t.cdf((actual + 0.5 - mu) / sigma, df) - stats.t.cdf((actual - 0.5 - mu) / sigma, df)
    return -np.sum(np.log(np.clip(p, 1e-9, None)))


def fit_pooled(samples: List[Sample]) -> CalibParams:
    txn = np.array([s.txn for s in samples], float)
    xnd = np.array([s.xnd for s in samples], float)
    act = np.array([s.actual for s in samples], float)
    x0 = [float(np.mean(act - txn)), 0.0, np.log(1.5)]
    res = optimize.minimize(
        _nll, x0, args=(txn, xnd, act), method="L-BFGS-B",
        bounds=[(-8, 8), (np.log(0.3), np.log(4.0)), (np.log(0.3), np.log(6.0))],
    )
    b, ls, lf = res.x
    return CalibParams(float(b), float(np.exp(ls)), float(np.exp(lf)), DF)


@dataclass
class WeatherCalibration:
    pooled: Dict[str, CalibParams] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    station_bias: Dict[str, Dict[str, float]] = field(default_factory=dict)  # kind -> station -> bias
    station_n: Dict[str, Dict[str, int]] = field(default_factory=dict)
    n: Dict[str, int] = field(default_factory=dict)
    fitted_through: Optional[str] = None
    fitted_at: Optional[str] = None
    # Trust in the engine vs the market, measured out-of-sample by the backtest.
    # None/0 means no demonstrated edge: the engine prices markets but can't trade.
    weight: Optional[float] = None
    weight_source: Optional[str] = None

    def params(self, station: str, kind: str) -> CalibParams:
        base = self.pooled.get(kind) or DEFAULT_PARAMS[kind]
        bias = self.station_bias.get(kind, {}).get(station, base.bias)
        return CalibParams(bias, base.scale, base.floor, base.df)

    @classmethod
    def fit(cls, samples: Iterable[Sample], min_samples: int = 30) -> "WeatherCalibration":
        samples = list(samples)
        cal = cls()
        for kind in ("high", "low"):
            ks = [s for s in samples if s.kind == kind]
            if len(ks) < min_samples:
                continue
            pooled = fit_pooled(ks)
            by_station: Dict[str, List[float]] = {}
            for s in ks:
                by_station.setdefault(s.station, []).append(s.actual - s.txn)
            biases, counts = {}, {}
            for st, resid in by_station.items():
                n = len(resid)
                w = n / (n + STATION_PRIOR_N)
                biases[st] = w * float(np.mean(resid)) + (1 - w) * pooled.bias
                counts[st] = n
            # Second pass: spread fit on bias-corrected forecasts, so station
            # quirks don't masquerade as forecast uncertainty.
            adjusted = [Sample(s.station, kind, s.target, s.txn + biases[s.station], s.xnd, s.actual) for s in ks]
            spread = fit_pooled(adjusted)
            cal.pooled[kind] = CalibParams(round(pooled.bias, 3), round(spread.scale, 3), round(spread.floor, 3), DF)
            cal.station_bias[kind] = {st: round(b + spread.bias, 3) for st, b in biases.items()}
            cal.station_n[kind] = counts
            cal.n[kind] = len(ks)
        if samples:
            cal.fitted_through = max(s.target for s in samples).isoformat()
        cal.fitted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return cal

    # -- persistence -----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "pooled": {k: asdict(v) for k, v in self.pooled.items()},
            "station_bias": self.station_bias,
            "station_n": self.station_n,
            "n": self.n,
            "fitted_through": self.fitted_through,
            "fitted_at": self.fitted_at,
            "weight": self.weight,
            "weight_source": self.weight_source,
        }

    def save(self, path: Path | str = DEFAULT_CALIBRATION_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CALIBRATION_PATH) -> "WeatherCalibration":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def from_dict(cls, d: dict) -> "WeatherCalibration":
        cal = cls()
        cal.pooled.update({k: CalibParams(**v) for k, v in d.get("pooled", {}).items()})
        cal.station_bias = d.get("station_bias", {})
        cal.station_n = d.get("station_n", {})
        cal.n = d.get("n", {})
        cal.fitted_through = d.get("fitted_through")
        cal.fitted_at = d.get("fitted_at")
        cal.weight = d.get("weight")
        cal.weight_source = d.get("weight_source")
        return cal
