"""
CI regression gate for RAGAS metrics.

Compares current eval metrics against a stored baseline.
Fails (raises RegressionError) if any metric drops > threshold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import structlog

log = structlog.get_logger()

TRACKED_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "aggregate"]


class RegressionError(Exception):
    """Raised when a metric drops beyond the allowed threshold."""


@dataclass
class RegressionReport:
    """Result of a regression check."""

    passed: bool
    details: dict[str, dict[str, float]]  # metric -> {baseline, current, delta, threshold}
    failures: list[str]  # list of failing metric names


def load_baseline(baseline_path: str) -> dict[str, float]:
    """Load stored baseline metrics from JSON file."""
    path = Path(baseline_path)
    if not path.exists():
        log.warning("regression_gate.no_baseline", path=baseline_path)
        return {}
    with open(path) as f:
        return json.load(f)


def save_baseline(metrics: dict[str, float], baseline_path: str) -> None:
    """Save current metrics as the new baseline."""
    path = Path(baseline_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("regression_gate.baseline_saved", path=baseline_path, metrics=metrics)


def check_regression(
    current_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    threshold: float = 0.05,
) -> RegressionReport:
    """
    Compare current metrics against baseline.

    Args:
        current_metrics: Metrics from the current evaluation run.
        baseline_metrics: Previously saved baseline metrics.
        threshold: Maximum allowed relative drop (default 5%).

    Returns:
        RegressionReport with pass/fail status and per-metric details.
    """
    if not baseline_metrics:
        log.info("regression_gate.no_baseline_skipping_check")
        return RegressionReport(passed=True, details={}, failures=[])

    details: dict[str, dict[str, float]] = {}
    failures: list[str] = []

    for metric in TRACKED_METRICS:
        if metric not in baseline_metrics or metric not in current_metrics:
            continue

        baseline_val = baseline_metrics[metric]
        current_val = current_metrics[metric]

        if baseline_val == 0.0:
            delta = 0.0
        else:
            delta = (baseline_val - current_val) / baseline_val  # positive = regression

        details[metric] = {
            "baseline": round(baseline_val, 4),
            "current": round(current_val, 4),
            "delta": round(delta, 4),
            "threshold": threshold,
        }

        if delta > threshold:
            failures.append(metric)
            log.error(
                "regression_gate.metric_failed",
                metric=metric,
                baseline=baseline_val,
                current=current_val,
                drop_pct=round(delta * 100, 2),
                threshold_pct=round(threshold * 100, 2),
            )
        else:
            log.info(
                "regression_gate.metric_passed",
                metric=metric,
                baseline=baseline_val,
                current=current_val,
                drop_pct=round(delta * 100, 2),
            )

    report = RegressionReport(passed=len(failures) == 0, details=details, failures=failures)
    return report
