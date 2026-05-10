"""Generate an Evidently drift report against the baseline distribution.

Compares a 'current' snapshot of input data against the training baseline
(captured by scripts/capture_baseline.py). Outputs:

    reports/drift/<timestamp>_drift_report.html  : full interactive HTML
    reports/drift/<timestamp>_summary.json       : machine-readable summary
    reports/drift/latest.html                    : symlink to most recent
    reports/drift/latest_summary.json            : symlink to most recent

Usage:

    # Synthetic comparison (good for demos): perturbs the baseline by drift_factor
    uv run python scripts/run_drift_report.py --drift-factor 0.3

    # Real-data comparison: point at a parquet of recent /predict inputs
    uv run python scripts/run_drift_report.py --current data/processed/recent_predictions.parquet

    # Severe drift simulation (for testing alerting thresholds)
    uv run python scripts/run_drift_report.py --drift-factor 1.5
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT_DIR, get_logger

logger = get_logger("drift_report")

DEFAULT_BASELINE = ROOT_DIR / "data" / "processed" / "drift_baseline.parquet"
REPORTS_DIR = ROOT_DIR / "reports" / "drift"


def synthetic_current_data(
    baseline: pd.DataFrame,
    drift_factor: float = 0.0,
    n_rows: int = 1_000,
    seed: int = 99,
) -> pd.DataFrame:
    """Generate a 'current' dataset by perturbing the baseline.

    drift_factor=0.0 → resample baseline (drift score should be ~0)
    drift_factor=0.3 → moderate shift on top numeric features
    drift_factor=1.0 → strong shift, most numeric features will drift
    drift_factor=1.5 → severe drift, useful for testing alerting

    This is a stand-in for real production data. In a deployed system, the
    'current' data would be captured from /predict logs (S3 parquet, daily
    rollup, etc.) and fed in via --current.
    """
    current = baseline.sample(n=min(n_rows, len(baseline)), random_state=seed).reset_index(drop=True)

    if drift_factor > 0:
        # Shift numeric columns to simulate input drift. We perturb the first
        # several numeric columns by drift_factor × stdev — this is a
        # well-defined synthetic drift that Evidently's KS-test will detect.
        numeric_cols = current.select_dtypes(include=["int64", "float64"]).columns
        n_to_perturb = min(8, len(numeric_cols))
        for col in numeric_cols[:n_to_perturb]:
            std = current[col].std()
            if pd.notna(std) and std > 0:
                current[col] = current[col] + drift_factor * std * 1.5

    return current


def extract_drift_metrics(report_dict: dict) -> dict:
    """Pull the headline drift numbers out of Evidently's report dict.

    Evidently's structure varies slightly across versions, so we look at all
    metrics and pick the first DataDrift / DatasetDrift one we find.
    """
    drift_score = None
    n_drifted_features = None
    n_total_features = None

    for m in report_dict.get("metrics", []):
        metric_name = m.get("metric", "")
        if "DatasetDrift" in metric_name or "DataDrift" in metric_name:
            result = m.get("result", {})
            if "share_of_drifted_columns" in result:
                drift_score = float(result["share_of_drifted_columns"])
            if "number_of_drifted_columns" in result:
                n_drifted_features = int(result["number_of_drifted_columns"])
            if "number_of_columns" in result:
                n_total_features = int(result["number_of_columns"])
            if drift_score is not None:
                break

    return {
        "drift_score": drift_score,
        "n_drifted_features": n_drifted_features,
        "n_total_features": n_total_features,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="Path to baseline parquet (default: data/processed/drift_baseline.parquet)",
    )
    parser.add_argument(
        "--current",
        default=None,
        help="Path to current data parquet. If omitted, synthetic data is generated.",
    )
    parser.add_argument(
        "--drift-factor",
        type=float,
        default=0.0,
        help="If --current is omitted, severity of synthetic drift (0=none, 1=strong)",
    )
    parser.add_argument(
        "--n-current",
        type=int,
        default=1_000,
        help="Number of synthetic 'current' rows to generate (default: 1,000)",
    )
    args = parser.parse_args()

    # ── Lazy import: Evidently is in the training extras only ───────────────
    try:
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.report import Report
    except ImportError as e:
        logger.error(
            "Evidently isn't installed. Run `uv sync --extra training` to install it."
        )
        raise SystemExit(1) from e

    # ── Load baseline ───────────────────────────────────────────────────────
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        logger.error(f"Baseline not found at {baseline_path}.")
        logger.error("Run `python scripts/capture_baseline.py` first.")
        sys.exit(1)

    baseline = pd.read_parquet(baseline_path)
    logger.info(f"Baseline: {len(baseline):,} rows × {baseline.shape[1]} cols")

    # ── Load or generate current data ───────────────────────────────────────
    if args.current and Path(args.current).exists():
        current = pd.read_parquet(args.current)
        source = f"file:{args.current}"
        logger.info(f"Current:  {len(current):,} rows from {args.current}")
    else:
        if args.current:
            logger.warning(
                f"--current path {args.current} not found. Falling back to synthetic data."
            )
        current = synthetic_current_data(
            baseline,
            drift_factor=args.drift_factor,
            n_rows=args.n_current,
        )
        source = f"synthetic:drift_factor={args.drift_factor}"
        logger.info(
            f"Current:  {len(current):,} rows synthetic ({source})"
        )

    # ── Run Evidently ───────────────────────────────────────────────────────
    logger.info("Running Evidently report (DataDrift + DataQuality)...")
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=baseline, current_data=current)

    # ── Save outputs ────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    html_path = REPORTS_DIR / f"{timestamp}_drift_report.html"
    report.save_html(str(html_path))
    size_kb = html_path.stat().st_size / 1024
    logger.info(f"HTML report     → {html_path} ({size_kb:.0f} KB)")

    drift_metrics = extract_drift_metrics(report.as_dict())
    summary = {
        "timestamp_utc": timestamp,
        "baseline_path": str(baseline_path.relative_to(ROOT_DIR)),
        "baseline_rows": len(baseline),
        "current_source": source,
        "current_rows": len(current),
        **drift_metrics,
        "html_report": str(html_path.relative_to(ROOT_DIR)),
    }

    summary_path = REPORTS_DIR / f"{timestamp}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON    → {summary_path}")

    # ── Update "latest" pointers (regular file copies, not symlinks — work on Windows) ─
    latest_html = REPORTS_DIR / "latest.html"
    latest_summary = REPORTS_DIR / "latest_summary.json"
    shutil.copyfile(html_path, latest_html)
    shutil.copyfile(summary_path, latest_summary)
    logger.info(f"Updated latest.html and latest_summary.json")

    # ── Print summary to stdout ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Drift report summary")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print("=" * 70)

    # Exit code reflects drift severity — useful for CI thresholds later.
    drift_score = summary.get("drift_score")
    if drift_score is not None and drift_score > 0.5:
        logger.warning(f"⚠ Significant drift detected: {drift_score:.1%} of features drifted")
    elif drift_score is not None:
        logger.info(f"Drift score: {drift_score:.1%} of features drifted")


if __name__ == "__main__":
    main()
