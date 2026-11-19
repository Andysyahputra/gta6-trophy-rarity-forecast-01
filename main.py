"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

Model: a trophy's printed rate is earners / players-tracked, and the two move
differently.  Exophase refreshes its counts on its own schedule, so between
refreshes every read is identical; only distinct snapshots carry information.
For each case we

  1. collapse the history to distinct snapshots (a new players-tracked count
     or any new rate), keeping the first read that shows each snapshot;
  2. fit log players-tracked against log time since release, and extrapolate
     it to the target;
  3. per trophy, recover earners (rate * players / 100) at each snapshot and
     fit log earners against log time since release, shrinking each trophy's
     slope toward the median slope of its tier (platinum / gold / silver /
     bronze) so noisy rare trophies borrow strength from their tier.  Both
     fits weight recent snapshots more (24-hour half-life), so the launch-night
     regime does not steer the extrapolation, and the log-earners fit adds a
     half-earner continuity offset that is subtracted back when forecasting,
     so small earner counts are not biased upward;
  4. forecast the rate as forecast earners / forecast players, and spread the
     five quantiles on the logit scale with a sigma that grows with the
     horizon and with the slope fit's own uncertainty.

When the history is a single frozen snapshot (the practice cases), every fit
is flat, so the forecast is the last rate with a spread that still admits a
refresh jump.  Pure standard library, deterministic: same input bytes give
the same output bytes.

Run: python3 main.py, reading $MAKEFUN_INPUT_DIR (else /input) and writing
$MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Spread of the logit quantiles: a floor that admits a refresh re-basing the
# page even over a short horizon, plus a term that grows with the horizon in
# days, plus each trophy's own slope uncertainty (added per trophy below).
SIGMA_FLOOR = 0.30
SIGMA_PER_DAY = 0.10
SIGMA_CAP = 1.60

# How strongly a trophy's fitted log-earners slope is pulled toward its
# tier's median slope: pseudo-count of tier observations.
SHRINK_PRIOR = 2.0


def parse_time(text):
    """ISO 8601 (with Z or offset, date or datetime) -> aware datetime UTC."""
    text = str(text).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = math.floor(x * 10000.0 + 0.5) / 10000.0
    return int(scaled) if float(scaled).is_integer() else scaled


def solve3(mat, vec):
    """Solve a 3x3 linear system by Gaussian elimination; None if singular."""
    rows = [row[:] + [v] for row, v in zip(mat, vec)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(rows[r][col]))
        if abs(rows[piv][col]) < 1e-12:
            return None
        rows[col], rows[piv] = rows[piv], rows[col]
        for r in range(col + 1, 3):
            f = rows[r][col] / rows[col][col]
            for c in range(col, 4):
                rows[r][c] -= f * rows[col][c]
    out = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        s = rows[r][3] - sum(rows[r][c] * out[c] for c in range(r + 1, 3))
        out[r] = s / rows[r][r]
    return out


def quad_fit(xs, ys, ws):
    """Weighted fit y ~ a + b*x + c*x^2. Returns (a, b, c) with c <= 0, or
    None when there are too few points, the system is singular, or the
    fitted curve is not concave (then the linear fit stands)."""
    if len(xs) < 5:
        return None
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    # Centre x for numerical stability; the returned a is at the centre.
    xc = [x - mx for x in xs]
    s = [0.0] * 5
    t = [0.0] * 3
    for x, y, w in zip(xc, ys, ws):
        p = 1.0
        for k in range(5):
            s[k] += w * p
            p *= x
        p = 1.0
        for k in range(3):
            t[k] += w * p * y
            p *= x
    mat = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]]
    sol = solve3(mat, t)
    if sol is None:
        return None
    a, b, c = sol
    if c > 0:
        return None
    return (a, b, c, mx)


def ols(xs, ys, ws=None):
    """Least squares y ~ a + b*x, optionally recency-weighted. Returns
    (a, b, se_b, n); se_b is 0 when the fit is exact, degenerate, or has too
    few points to judge. Weights default to 1."""
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0, 0)
    if ws is None:
        ws = [1.0] * n
    if n == 1:
        return (ys[0], 0.0, 0.0, 1)
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return (my, 0.0, 0.0, n)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    se_b = 0.0
    if n > 2:
        resid2 = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        dof = n - 2
        if resid2 > 0:
            se_b = math.sqrt((resid2 / dof) / sxx)
    return (a, b, se_b, n)


def median(values):
    ordered = sorted(values)
    m = len(ordered)
    if m == 0:
        return 0.0
    if m % 2:
        return ordered[m // 2]
    return 0.5 * (ordered[m // 2 - 1] + ordered[m // 2])


def distinct_snapshots(case):
    """Collapse reads that repeat the previous snapshot exactly. A snapshot
    is (time, players_tracked, rates tuple); the first read showing it is
    kept, later identical reads carry no new information."""
    snaps = []
    prev_key = None
    for read in case["history"]:
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        if not players or players <= 0:
            continue
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        snaps.append((parse_time(read["at"]), players, rates))
    return snaps


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    release = parse_time(case["game"]["release_date"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    def log_time(moment):
        hours = (moment - release).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    x_target = log_time(target)
    snaps = distinct_snapshots(case)
    if not snaps:
        # No usable history: flat 1% medians with a wide spread.
        return {"trophies": [
            {"id": t["id"], "q": quantiles(1.0, SIGMA_CAP)} for t in trophies
        ]}

    xs = [log_time(at) for at, _, _ in snaps]
    x_last = xs[-1]
    # Recency weights: a snapshot a day old counts half as much as the last
    # one, so the fit follows the current regime, not launch night.
    times = [at for at, _, _ in snaps]
    last_time = times[-1]
    weights = [0.5 ** ((last_time - at).total_seconds() / 86400.0)
               for at in times]

    # --- players-tracked growth -------------------------------------------
    log_players = [math.log(p) for _, p, _ in snaps]
    a_p, b_p, _, _ = ols(xs, log_players, weights)
    players_star = math.exp(a_p + b_p * x_target)
    players_star = max(players_star, snaps[-1][1])

    # --- per-trophy earners growth ---------------------------------------
    # First pass: raw fits (and the per-trophy points, kept for re-anchoring
    # after shrinkage), to find each tier's median slope.
    series = []
    for i, trophy in enumerate(trophies):
        ex, ey, ew = [], [], []
        for (at, players, rates), x, w in zip(snaps, xs, weights):
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            earners = rate * players / 100.0
            ex.append(x)
            ey.append(math.log(earners + 0.5))
            ew.append(w)
        series.append((ex, ey, ew))
    raw = [ols(ex, ey, ew) for ex, ey, ew in series]
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = [raw[i][1] for i, t in enumerate(trophies)
                  if t.get("tier") == tier and raw[i][3] >= 2]
        if slopes:
            tier_slopes[tier] = median(slopes)
    all_slopes = [r[1] for r in raw if r[3] >= 2]
    fallback_slope = median(all_slopes) if all_slopes else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        a_i, b_i, se_b, n_i = raw[i]
        tier = trophy.get("tier")
        prior = tier_slopes.get(tier, fallback_slope)
        # Shrink the slope toward the tier median; a trophy whose earners
        # never moved keeps a slope near its tier's, not exactly zero.
        b_shrunk = (n_i * b_i + SHRINK_PRIOR * prior) / (n_i + SHRINK_PRIOR)
        # Re-anchor the intercept so the shrunk line still passes through
        # the fit's centre (weighted mean x, weighted mean y of this
        # trophy's points).
        ex, ey, ew = series[i]
        if ex:
            sw = sum(ew)
            mx = sum(w * x for w, x in zip(ew, ex)) / sw
            my = sum(w * y for w, y in zip(ew, ey)) / sw
            a_shrunk = my - b_shrunk * mx
        else:
            a_shrunk = math.log(0.5)
        # Undo the half-earner continuity offset: the fit was on
        # log(earners + 0.5), so the forecast earners are exp(...) - 0.5.
        earners_star = math.exp(a_shrunk + b_shrunk * x_target) - 0.5
        earners_star = min(max(earners_star, 0.0), players_star)
        # Earners growth decelerates as a trophy's cohort saturates, so a
        # concave quadratic in log time may damp the linear extrapolation,
        # never raise it.
        quad = quad_fit(ex, ey, ew)
        if quad is not None:
            qa, qb, qc, qmx = quad
            dxq = x_target - qmx
            earners_quad = math.exp(qa + qb * dxq + qc * dxq * dxq) - 0.5
            earners_star = min(earners_star, max(earners_quad, 0.0))
        rate_star = 100.0 * earners_star / players_star if players_star else 0.01

        # Spread: floor + horizon growth + slope uncertainty over the
        # extrapolation distance in log time.
        dx = max(0.0, x_target - x_last)
        sigma = math.sqrt(
            SIGMA_FLOOR ** 2
            + SIGMA_PER_DAY * horizon_days
            + (se_b * dx) ** 2
        )
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        rows.append({"id": trophy["id"],
                     "q": quantiles(rate_star, sigma)})
    return {"trophies": rows}


def quantiles(rate_star, sigma):
    centre = logit(rate_star)
    qs = [round4(inv_logit(centre + z * sigma)) for z in Z_SCORES]
    # Rounding can only tie neighbours, never invert them; enforce anyway.
    for j in range(1, len(qs)):
        if qs[j] < qs[j - 1]:
            qs[j] = qs[j - 1]
    return qs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
