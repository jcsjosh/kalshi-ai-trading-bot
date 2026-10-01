"""Edge engines — model-driven probability sources with a shared, fee-aware evaluator.

Each engine answers one question for the markets it understands: *what is the
true probability this contract pays?* The evaluator then turns that number and
the live book into a decision (take, post, or pass) after Kalshi's real fees,
shrinking the engine's view toward the market by however much skill the engine
has actually demonstrated out-of-sample. Engines never place orders themselves;
execution goes through ``src.agent.toolbelt.place_guarded_order`` so every
trade passes the risk governor and Edge Policy, and lands in the decision
journal (tagged ``method=engine:<name>``) where ``cli.py edge`` scores it and
``cli.py improve`` can block a losing engine.

Engines:

* ``weather``   - NOAA National Blend of Models station forecasts, calibrated
                   per station against the exact NWS climate reports Kalshi
                   settles on, truncated by today's observations.
* ``arbitrage`` - structural mispricings (mutually exclusive baskets, proven-
                   exhaustive buckets, same-subject strike ladders) priced off
                   live order book depth and per-series fees.

See ``docs/ENGINES.md`` for the measured results.
"""

ENGINE_NAMES = ("weather", "arbitrage")
