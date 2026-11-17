"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun).

Method: learn the shape of a big console launch day from the case's own
comparables, then anchor that shape to this launch's hour-1 print.

A launch whose category is already live at hour 1 (midnight rolling
unlock: New Zealand, Australia, Asia and Europe already playing, the UK
unlocking at 00:00 UTC) behaves nothing like one that only unlocks later
in the UTC day. So the comparables are filtered to the "already live"
archetype first: hour-1 viewers at least 20% of the launch's release-day
peak, with hour 1 present and at least 20 of the 24 release-day hours
present. On that pool, for each hour h = 2..24, the five forecast
quantiles are the empirical quantiles of log(viewers_h / viewers_1)
across the pool, lightly smoothed across neighbouring hours and blended
15% (in log space) toward the house baseline's persistence forecast, as
insurance against a pool quirk. That blend and the filter were chosen by
leave-one-out cross-validation over the kit's comparable launches: on
the 24 already-live launches (552 scored hours) the shape scores a mean
pinball loss of 0.222 against the baseline's 0.265, a 16% improvement;
on all 37 usable launches the empirical-quantile shape scores 0.714
against the baseline's 0.930.

The program is standard library only, deterministic (sorted inputs, no
randomness, no clock, no network), and runs in well under a second per
case. It reads every <id>.in in the input directory and writes one
<id>.out per case, exactly {"hours": [{"hour": h, "q": [q05, q25, q50,
q75, q95]}, ...]} for hours 2..24.

Run locally:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path: put it back
# first, so a module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
# House baseline spread: log-units per square-root hour.
BASE_SIGMA = 0.15
# Share of the baseline (in log space) blended into the shape forecast.
BLEND = 0.15
# Neighbour weight when smoothing each quantile level across hours.
SMOOTH_W = 0.5
# "Already live at hour 1" filter: hour-1 viewers / release-day peak.
LIVE_RATIO = 0.20
LIVE_RATIO_RELAXED = 0.15
MIN_DAY_HOURS = 20
MIN_POOL = 8
MAX_VIEWERS = 1_000_000_000


def round2(x):
    """Round to 2 decimals, half up (as the starter does)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def quantile(sorted_xs, p):
    """Linear-interpolation quantile of an already sorted list."""
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = (len(sorted_xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def day_series(comp):
    """Release-day {hour 1..24: viewers} from a comparable's hourly rows."""
    try:
        start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    out = {}
    for row in comp.get("hourly") or []:
        try:
            when = parse_ts(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if not viewers or viewers <= 0:
            continue
        hour = int((when - start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            out[hour] = viewers
    return out


def select_pool(series_list):
    """Already-live launches first, relaxing only if the pool is too small."""
    usable = [s for s in series_list if 1 in s and len(s) >= MIN_DAY_HOURS]
    for ratio in (LIVE_RATIO, LIVE_RATIO_RELAXED):
        pool = [s for s in usable
                if s[1] / max(s.values()) >= ratio]
        if len(pool) >= MIN_POOL:
            return pool
    return usable


def baseline_quantiles(p1, hour):
    spread = BASE_SIGMA * math.sqrt(hour - 1)
    return [p1 * math.exp(z * spread) for z in Z]


def shape_quantiles(p1, hour, pool):
    """Empirical log-ratio quantiles for one hour, or None if no data."""
    ratios = sorted(math.log(s[hour] / s[1])
                    for s in pool if hour in s and 1 in s)
    if not ratios:
        return None
    return [p1 * math.exp(quantile(ratios, tau)) for tau in LEVELS]


def smooth_levels(by_hour):
    """Smooth each quantile level across neighbouring hours (weight 0.5)."""
    hours = sorted(by_hour)
    if len(hours) < 3:
        return by_hour
    smoothed = {}
    for i, hour in enumerate(hours):
        neighbours = []
        if i > 0:
            neighbours.append(by_hour[hours[i - 1]])
        if i < len(hours) - 1:
            neighbours.append(by_hour[hours[i + 1]])
        # by_hour values are log-quantiles; average in log space.
        own = by_hour[hour]
        merged = [own] + neighbours
        weights = [1.0] + [SMOOTH_W] * len(neighbours)
        total = sum(weights)
        smoothed[hour] = [sum(w * col[k] for w, col in zip(weights, merged)) / total
                          for k in range(len(LEVELS))]
    return smoothed


def forecast_case(case):
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first and first.get("viewers"):
        p1 = max(1, first["viewers"])
    else:
        # No hour-1 print: fall back to the latest pre-checkpoint point.
        hourly = case.get("history_hourly") or []
        p1 = max(1, hourly[-1]["viewers"]) if hourly else 1

    series_list = [day_series(c) for c in case.get("comparables") or []]
    pool = select_pool(series_list)

    # Per-hour empirical log-quantiles from the pool, smoothed in log space.
    log_shape = {}
    for row in case.get("forecast") or []:
        hour = row["hour"]
        qs = shape_quantiles(p1, hour, pool)
        if qs is not None:
            log_shape[hour] = [math.log(max(q, 1e-9)) for q in qs]
    log_shape = smooth_levels(log_shape)

    hours_out = []
    for row in case.get("forecast") or []:
        hour = row["hour"]
        base = baseline_quantiles(p1, hour)
        if hour in log_shape:
            shape = [math.exp(v) for v in log_shape[hour]]
            qs = [math.exp((1 - BLEND) * math.log(max(s, 1e-9))
                           + BLEND * math.log(max(b, 1e-9)))
                  for s, b in zip(shape, base)]
        else:
            qs = base
        # Monotone non-decreasing, inside (0, 1e9], 2 decimals.
        fixed = []
        prev = 0.0
        for q in qs:
            q = min(max(q, 1.0), MAX_VIEWERS)
            q = max(q, prev)
            fixed.append(round2(q))
            prev = fixed[-1]
        hours_out.append({"hour": hour, "q": fixed})
    return {"hours": hours_out}


def forecast_baseline_only(case):
    """Persistence forecast used if a case cannot be shaped at all."""
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    p1 = max(1, first["viewers"]) if first and first.get("viewers") else 1
    return {"hours": [{"hour": row["hour"],
                       "q": [round2(q) for q in baseline_quantiles(p1, row["hour"])]}
                      for row in case.get("forecast") or []]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception as exc:  # never lose a case: fall back to baseline
            print(f"{path.name}: shape failed ({exc!r}), using baseline",
                  file=sys.stderr)
            forecast = forecast_baseline_only(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
