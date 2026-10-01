# Edge Engines

Model-driven probability sources that plug into this toolkit's guard stack. An
engine prices markets from its own model; a shared evaluator turns that price
and the live book into a fee-aware, Kelly-sized decision; and every live order
goes through `place_guarded_order`, the same risk governor, Edge Policy, size
caps and decision journal as everything else. Engines get no private path to
the exchange.

The design goal is the repo's thesis applied to models: **an engine trades only
as far as its out-of-sample record has earned.** The first engine shipped here
failed that test, and the system refuses to trade it. That is the point.

```bash
python cli.py engines scan                 # price everything, list opportunities (read-only, no key)
python cli.py engines run                  # same, then record to the paper ledger (nothing sent)
python cli.py engines paper --settle       # attach settlements, score each engine vs the market
python cli.py engines backtest --save-weight   # walk-forward weather backtest -> earned trust weight
python cli.py engines calibrate            # refit weather calibration on recent NOAA vs CLI data
python cli.py engines run --live           # REAL orders, through the governor + Edge Policy
```

The MCP server exposes `engines_scan` and `engines_paper` (read-only).

---

## How a decision is made

1. **Engine view.** The engine returns `p_yes` for each market it understands,
   plus an uncertainty (`stderr`) and its *trust weight* `w`.
2. **Shrink toward the market.** `logit(p_fair) = w * logit(p_engine) + (1 - w) * logit(p_market)`,
   where `p_market` is the de-vigged mid. A liquid Kalshi book is usually sharp,
   so an engine with no demonstrated edge has `w = 0` and `p_fair` *is* the market.
3. **Price every entry after real fees.** Fees come from each series' own
   `fee_type` / `fee_multiplier` (`GET /series/{t}`): taker
   `ceil(M x 0.07 x C x P x (1-P))`, maker `ceil(M x 0.0175 x ...)` only on
   maker-fee series. Taker entries are valued at `p_fair`. **Maker entries are
   valued as if filled:** a resting bid fills when someone sells to you at your
   price, so on a fill the market's estimate is your price, not the mid. Only
   the engine's own disagreement with the posted price counts as edge. (Without
   this, an uninformed engine "finds edge" by bidding inside every wide spread.)
4. **Margin grows with uncertainty:** `ev >= min_edge + z * stderr` (default 3c).
5. **Fractional Kelly** (0.25x), capped at 3% of bankroll per market, 6% per
   event, 30% per scan, and by visible depth. Snapshot quotes only nominate;
   every nominee is re-priced off its live order book before sizing.
6. **Execution** (live only): `place_guarded_order` with `method=engine:<name>`,
   taker as immediate-or-cancel, maker as an expiring limit. The journal tag
   means `cli.py learnings` / `improve` score each engine separately, and the
   Edge Policy blocks an engine with 5+ settled losing trades automatically.

## Engine: weather

Kalshi lists daily high and low temperature markets for 24 US stations (about
48 active series, 6 mutually exclusive 2-degree buckets per day). They settle
on the NWS Daily Climate Report (CLI) for one station, named in the rules.

**Model.** NOAA's National Blend of Models station bulletin (`NBS`, via the IEM
archive) gives the forecast max/min (`txn`) and NOAA's own spread (`xnd`) for
the exact settlement station. The predictive distribution is a Student-t on the
whole-degree outcome:

```
mu    = txn + station_bias          sigma = sqrt((scale * xnd)^2 + floor^2)
P(T = v) = F(v + 0.5) - F(v - 0.5)  (the CLI reports whole degrees)
```

`bias`, `scale` and `floor` are fit by maximum likelihood per kind (high/low),
with each station's bias shrunk toward the pool (`n / (n + 15)`), on NBM
forecasts vs IEM's archive of the actual CLI reports, using only runs that were
published (run time + 2h) before the decision. Same-day markets are truncated by
what has already been observed (NWS station observations; QC-rejected readings
and isolated spikes are dropped), and an outcome the observations have already
ruled out is priced at certainty, independent of forecast skill.

**Facts the source weather bot got wrong** (and this engine verifies against the
live API): `B75.5` is the 75-76 bucket (`strike_type: between`), not "above
75.5"; NYC settles at Central Park, Chicago at Midway, Houston at Hobby, DC at
Reagan National; the CLI day is midnight to midnight *local standard time*
(1 AM to 1 AM in summer), not the UTC day.

### Measured result: no edge

Walk-forward backtest, 2026-08-03 to 2026-09-30 (calibration history from
2026-04-01), every settled temperature event Kalshi still serves candles for:
**2,180 events, 19,620 bucket decisions**, decisions at 20:00 UTC the day before
and 14:00 UTC on the day (highs), buying at the hourly candle's ask with exact
fees. Full report: [`ENGINES_WEATHER_BACKTEST.md`](ENGINES_WEATHER_BACKTEST.md).

| Forecast quality (out-of-sample) | Brier | Log loss |
|---|---|---|
| Kalshi market (de-vigged mid) | **0.1037** | **0.3262** |
| Engine (calibrated NBM) | 0.1189 | 0.3729 |

The market beats calibrated NBM in nearly every series. Blending the engine in
improves log loss only marginally (z = 1.73 over the whole window, below the
z >= 2 bar), so the **earned trust weight is 0**.

Simulated trading with the walk-forward weight lost money anyway, and the way it
lost is the useful lesson:

| | Trades | P&L | ROI | t-stat |
|---|---|---|---|---|
| Engine alone (walk-forward weight) | 330 | **-$888** | -12.4% | -2.2 |
| With the Edge Policy in the loop | 8 | **-$68** | | blocked 2026-08-23 |

In early September the blend *did* beat the market's log loss significantly
(z ~ 2.5-2.8) across all buckets. But trades come from the buckets where the
model disagrees with the market most (about 25 points on median), and there the
market was right: the engine predicted a 56% win rate on its trades and won
44.8%. **A model can be slightly informative on average and still be worst
exactly where it bets** (the winner's curse). So the only honest gate is the
engine's own out-of-sample *trading* record: `--save-weight` writes `w = 0`
whenever simulated trading lost, and the live Edge Policy, replayed over the
same trades, would have shut the engine off after 8 trades.

What the live engine does today: prices all ~576 open temperature markets,
reports its view next to the market's, and only trades outcomes that today's
observations have already decided (when those still trade at least 3c away from
certainty after fees, which is rare).

## Engine: arbitrage

Baskets whose worst-case payout beats their fee-inclusive cost, so no
forecasting is involved:

* **all-NO** on a mutually exclusive event: at most one leg resolves YES.
* **all-YES** on an event whose strikes are *proven* to tile every possible
  outcome. Only whole-number outcomes qualify (the NWS temperature buckets). A
  cent-grid tiling is not proof: an ETH bucket set settling on a 60-second
  average can land between 1249.99 and 1250.00.
* **ladders** on the same subject: P(above X1) >= P(above X2) for X1 < X2.

Snapshot prices nominate; each basket is then walked off live depth to the
count that maximizes *total* guaranteed profit (marginal baskets that lose
money individually are not bought). Live execution preflights every leg through
the guard stack before sending any (a policy refusal on leg 3 after legs 1-2
filled would leave an unhedged position), sends the scarcest leg first as
immediate-or-cancel, and shrinks later legs to what filled.

**Data traps this engine hit and now rejects:**

* A Kalshi game event holds ladders on *different* quantities (every player's
  receiving yards, each team's margin). Pairing across them produced ~24,000
  false "violations". Ladders now pair only markets with the same subject (the
  title with numbers masked).
* Kalshi shipped "exactly 5 Starship launches" as `strike_type: less` with
  `floor = cap = 5`. Read literally that is a $325 "risk-free" ladder. Strike
  metadata whose shape contradicts its type, or rules saying "exactly" on a
  one-sided type, is now treated as non-numeric.

**Measured result:** on 2026-10-01 a full scan of 13,842 open events and 135,000
markets took ~16 seconds and found no basket that survives fees and live depth
(the last candidate, ETH year-end buckets, was worth $0.17 and is not provably
exhaustive). Kalshi's books are internally consistent after fees almost all the
time. The engine is cheap insurance for the moments they aren't.

---

## Where these pieces came from

| Repo | Used for |
|---|---|
| `kalshi-ai-trading-bot` (this fork) | The base: risk governor, guarded orders, decision journal, edge / policy / learnings loop, MCP server. Engines plug into it rather than around it. |
| `prediction-market-analysis` (your Kalshi edge finder) | Fee formula and per-order rounding, fractional Kelly sizing, the all-NO / all-YES / strike-ladder basket taxonomy, and the maker-vs-taker calibration findings that motivated valuing maker fills conservatively. |
| `oracle3-prediction-market-agent` | The current Kalshi fee schedule (maker 0.0175 on maker-fee series, per-series `fee_multiplier`) and the constraint-relations framing for arbitrage. |
| `polymarket-kalshi-weather-bot` | The weather-market idea. Its bucket parsing, station choice and day boundary were wrong (above), and its GFS-at-city-center ensemble was replaced with NBM at the settlement station. |
| `pykalshi`, `kalshi-starter-code-python` | Cross-checks for the v2 single-book order semantics and request signing the fork's client already implements. |
| `mcp-server-kalshi` | The preview-unless-confirmed safety pattern (engines are dry by default) and the habit of reading the actual rules, which is how the mislabeled strikes were caught. |
| `KalshiMarketMaker` | Not integrated: continuous Avellaneda-Stoikov quoting needs a live, always-on runtime, and the fork already has `src/strategies/market_making.py`. Its lesson shows up in the maker-fill valuation. |
| `sports-skills`, `dr-manhattan`, `pmxt`, `prediction-market-mcp`, `simmer-sdk`, `CloddsBot` | Not integrated yet. Sportsbook de-vig and Polymarket consensus are the obvious next engines, and should face the same backtest gate (the fork's own 12-agent study found liquid sports books efficient). |
| `tools-and-analysis` | Kalshi's community notebooks (CPI surprise, Fed): consistent with the fork's record, which already blocks economic-data buckets. |
| `context-mode`, `orca`, `skills`, awesome lists | Developer tooling and directories, not trading code. Nothing to integrate. |

## Limits and next steps

* The weather backtest buys at hourly candle asks with an assumed 100 contracts
  of depth and simulates no maker fills. Kalshi only serves candles for markets
  settled after its historical cutoff (about two months of data in 2026-10).
* Forward evidence beats any backtest: run `cli.py engines run` (paper) every
  loop tick and `engines paper --settle` daily. An engine has earned live
  capital when its paper log loss beats the market's over many settled orders
  *and* its paper P&L is positive.
* Candidate engines, each to be shipped only behind the same gate: Polymarket
  consensus on matched markets, sportsbook de-vig on thin Kalshi sports markets,
  and maker quoting around a fair value with real fill tracking.
