#!/usr/bin/env python3
"""
CI script: compare latest eval results against stored baseline.

Usage:
  python scripts/check_regression.py --threshold 0.05

Exit code 0 = pass, 1 = regression detected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import structlog

from knowledge_engine.evaluation.regression_gate import check_regression, load_baseline

log = structlog.get_logger()

LATEST_METRICS_PATH = "results/latest_metrics.json"
BASELINE_PATH = "results/baseline_metrics.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check RAGAS regression")
    parser.add_argument("--threshold", type=float, default=0.05, help="Max allowed metric drop")
    parser.add_argument("--metrics", default=LATEST_METRICS_PATH, help="Latest metrics JSON path")
    parser.add_argument("--baseline", default=BASELINE_PATH, help="Baseline metrics JSON path")
    args = parser.parse_args()

    latest_path = Path(args.metrics)
    if not latest_path.exists():
        log.error("regression.no_latest_metrics", path=args.metrics)
        print(f"ERROR: Latest metrics not found at {args.metrics}")
        print("Run 'make eval' first to generate evaluation results.")
        return 1

    with open(latest_path) as f:
        current_metrics = json.load(f)

    baseline_metrics = load_baseline(args.baseline)

    report = check_regression(
        current_metrics=current_metrics,
        baseline_metrics=baseline_metrics,
        threshold=args.threshold,
    )

    print("\n=== Regression Gate Report ===")
    for metric, detail in report.details.items():
        status = "FAIL" if metric in report.failures else "PASS"
        print(
            f"  {status} {metric}: {detail['baseline']:.4f} -> "
            f"{detail['current']:.4f} "
            f"(delta: {detail['delta']*100:.2f}%, threshold: {detail['threshold']*100:.0f}%)"
        )

    if report.passed:
        print("\nAll metrics within threshold. Regression gate PASSED.")
        return 0
    else:
        print(f"\nMetrics failed: {', '.join(report.failures)}")
        print("Regression gate FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
