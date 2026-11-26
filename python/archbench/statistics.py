# SPDX-License-Identifier: AGPL-3.0-or-later
"""Descriptive statistics for retained per-call timing samples and modeled rates."""
from __future__ import annotations
from functools import lru_cache
import math
import random
import statistics


@lru_cache(maxsize=512)
def _median_interval_ranks(n: int, coverage: float = .95):
    """Central order-statistic interval; finite coverage needs at least six samples.

    Work down from the binomial center to avoid underflow of 2**(-n). The
    recurrence is linear or better in sample count and cached by n/coverage.
    """
    middle = n // 2
    probability = math.comb(n, middle) / (2 ** n)
    tail = .5 + (probability / 2 if n % 2 == 0 else 0)
    j = middle
    while j >= 0:
        if tail <= (1 - coverage) / 2:
            k = j + 1  # One-based lower order statistic.
            return k - 1, n - k, min(1., max(0., 1 - 2 * tail))
        tail -= probability
        probability *= j / (n - j + 1)
        j -= 1
    return None, None, 1.0  # Only the unbounded interval attains requested coverage.


def describe(samples, *, bootstrap=None):
    """Keep all samples; default CI uses distribution-free median order statistics.

    bootstrap=N explicitly requests N deterministic percentile-bootstrap draws;
    bootstrap=0 disables CI computation. A single sample has no estimated CV.
    """
    raw = list(samples)
    if not raw or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in raw): return {}
    ordered = [float(x) for x in raw]
    values = sorted(ordered)
    n = len(values)
    median, mean = statistics.median(values), statistics.mean(values)
    def quantile(p):
        position = (n - 1) * p
        lower = int(position)
        fraction = position - lower
        return values[lower] * (1 - fraction) + values[min(lower + 1, n - 1)] * fraction
    deviation = statistics.stdev(values) if n > 1 else None
    out = {"count": n, "median_ms": median, "median_us": median * 1000, "median_ns": median * 1e6,
           "mean_ms": mean, "min_ms": values[0], "max_ms": values[-1],
           "p10_batch_ms": quantile(.1), "p90_batch_ms": quantile(.9),
           "mad_ms": statistics.median(abs(x - median) for x in values),
           "stddev_ms": deviation, "cv": deviation / mean if deviation is not None else None,
           "relative_mad": statistics.median(abs(x - median) for x in values) / median,
           "range_fraction": (values[-1] - values[0]) / median}
    for field in ("mean", "min", "max", "mad", "stddev"):
        value = out[field + "_ms"]
        out[field + "_us"] = value * 1000 if value is not None else None
        out[field + "_ns"] = value * 1e6 if value is not None else None
    out["p10_batch_us"] = out["p10_batch_ms"] * 1000
    out["p90_batch_us"] = out["p90_batch_ms"] * 1000
    out["half_run_drift_fraction"] = None
    if n >= 4:
        half = n // 2
        out["half_run_drift_fraction"] = (statistics.median(ordered[-half:]) - statistics.median(ordered[:half])) / median
    if bootstrap is None:
        lower, upper, achieved = _median_interval_ranks(n)
        interval = [values[lower], values[upper]] if lower is not None else [None, None]
        out.update(median_ci95_ms=interval, median_ci_method="distribution_free_order_statistic",
                   median_ci_target_coverage=.95, median_ci_achieved_coverage=achieved,
                   median_ci_finite=lower is not None)
    elif bootstrap:
        if type(bootstrap) is not int or not 1 <= bootstrap <= 100000:
            raise ValueError("bootstrap must be an integer in 0..100000 or None")
        if n >= 3:
            rng = random.Random(20261125)
            medians = sorted(statistics.median(rng.choices(values, k=n)) for _ in range(bootstrap))
            out.update(median_ci95_ms=[medians[int(bootstrap * .025)], medians[min(bootstrap - 1, int(bootstrap * .975))]],
                       median_ci_method="percentile_bootstrap", median_ci_target_coverage=.95,
                       median_ci_achieved_coverage=None, median_ci_finite=True, bootstrap_resamples=bootstrap)
    if "median_ci95_ms" in out:
        interval = out["median_ci95_ms"]
        out["median_ci95_us"] = [v * 1000 if v is not None else None for v in interval]
        out["median_ci95_ns"] = [v * 1e6 if v is not None else None for v in interval]
        out["median_ci_relative_half_width"] = (interval[1] - interval[0]) / (2 * median) if all(v is not None for v in interval) else None
        out["median_ci_assumptions"] = "independent identically distributed samples; order-statistic coverage is conservative for ties; drift/correlation are not corrected"
        out["median_ci_coverage_scope"] = "nominal fixed-count coverage; adaptive stopping based on observed CV is not corrected"
    return out


def assess_quality(stats, *, target_cv=.05, minimum_samples=3):
    """Measurement quality is independent of correctness and hardware support."""
    result = {"label": "not_measured", "target_cv": target_cv, "observed_cv": stats.get("cv"),
              "scope": "retained within-run per-call sample averages; no filtering or guarantee of repeat-run stability",
              "reason": "no positive timing samples", "warnings": []}
    if not stats: return result
    if target_cv == 0:
        result.update(label="not_requested", reason="CV threshold disabled; raw dispersion is still reported")
    elif stats["count"] < minimum_samples or stats.get("cv") is None:
        result.update(label="insufficient_samples", reason=f"at least {minimum_samples} samples are needed to assess dispersion")
    elif stats["cv"] <= target_cv:
        result.update(label="stable", reason=f"sample CV <= {target_cv:g}")
    else:
        result.update(label="variable", reason=f"sample CV exceeds {target_cv:g}; retained samples were not discarded")
    drift = stats.get("half_run_drift_fraction")
    if drift is not None and abs(drift) > max(.05, 2 * target_cv):
        result["warnings"].append("first/last-half medians show drift; more samples do not remove systematic drift")
    if not stats.get("median_ci_finite", False):
        result["warnings"].append("a finite 95% median interval is unavailable at this sample count")
    result["half_run_drift_fraction"] = drift
    result["median_ci_relative_half_width"] = stats.get("median_ci_relative_half_width")
    return result


def rate_summary(samples_ms, count, *, scale=1e9, unit="Gop/s", timing_stats=None):
    """Rate of median time, sample rates, and reciprocal median-time CI.

    scale is units per displayed unit: 1e9 for giga, 1e12 for tera, 2**30 for GiB.
    Undefined/nonpositive work is unavailable, not zero throughput.
    """
    if type(count) not in (int, float) or not math.isfinite(count) or count <= 0: return None
    if not math.isfinite(scale) or scale <= 0: raise ValueError("rate scale must be positive")
    stats = timing_stats if timing_stats is not None else describe(samples_ms)
    if not stats: return None
    per_sample = [count / (value * .001 * scale) for value in samples_ms]
    sorted_rates = sorted(per_sample)
    def quantile(p):
        position = (len(sorted_rates) - 1) * p
        lower = int(position)
        return sorted_rates[lower] * (1 - (position - lower)) + sorted_rates[min(lower + 1, len(sorted_rates) - 1)] * (position - lower)
    interval = stats.get("median_ci95_ms", [None, None])
    finite_ci = len(interval) == 2 and all(type(v) in (int, float) and v > 0 for v in interval)
    ci = [count / (interval[1] * .001 * scale), count / (interval[0] * .001 * scale)] if finite_ci else [None, None]
    return {"unit": unit, "value": count / (stats["median_ms"] * .001 * scale),
            "statistic": "count divided by median per-call time", "samples": per_sample,
            "min": min(per_sample), "max": max(per_sample), "mean": statistics.mean(per_sample),
            "p10": quantile(.1), "p90": quantile(.9),
            "stddev": statistics.stdev(per_sample) if len(per_sample) > 1 else None,
            "ci95": ci, "ci_method": "reciprocal of median-time interval",
            "ci_time_method": stats.get("median_ci_method"),
            "ci_achieved_coverage": stats.get("median_ci_achieved_coverage")}
