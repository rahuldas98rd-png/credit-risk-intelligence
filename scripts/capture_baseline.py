"""Capture a baseline distribution snapshot from training data.

The 'baseline' is the reference distribution that drift reports compare
against. It defines what 'normal' looks like for subsequent drift detection.

Run this once per model training:
    uv run python scripts/capture_baseline.py

The output `data/processed/drift_baseline.parquet` is committed to the repo
so CI and local dev can run drift reports without the full training data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT_DIR, get_logger

logger = get_logger("capture_baseline")

DEFAULT_INPUT = ROOT_DIR / "data" / "raw" / "application_train.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "processed" / "drift_baseline.parquet"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to training CSV (default: data/raw/application_train.csv)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Where to write the baseline parquet",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=10_000,
        help="Number of rows to sample (default: 10,000 — enough for stable "
        "distribution stats, small enough to commit to git)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        logger.error("Pull the dataset from Kaggle first:")
        logger.error("  kaggle competitions download -c home-credit-default-risk -p data/raw")
        sys.exit(1)

    logger.info(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    n = min(args.n_rows, len(df))
    sample = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
    logger.info(f"Sampled {n:,} rows (seed={args.seed})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(output_path, index=False)

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Wrote baseline → {output_path} ({size_mb:.2f} MB)")
    logger.info(f"Run drift detection with: uv run python scripts/run_drift_report.py")


if __name__ == "__main__":
    main()
