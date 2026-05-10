"""CLI for MLflow Model Registry operations.

Usage:
    # First-time setup: register the existing trained model as v1, push to Staging
    python scripts/register_model.py --bootstrap

    # Promote the latest Staging version to Production (archives any existing Prod version)
    python scripts/register_model.py --promote

    # See current registry state
    python scripts/register_model.py --list

    # Combine: bootstrap + promote (e.g. for first-time setup)
    python scripts/register_model.py --bootstrap --promote
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to path so `from src.registry import ...` works when running
# this script directly from the repo root (`python scripts/register_model.py`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow

from src.registry import (
    MODEL_NAME,
    TRACKING_URI,
    bootstrap_register_existing,
    get_latest_version,
    transition_stage,
)


def cmd_bootstrap(name: str) -> int:
    """Register existing pickle files in models/ as a new registered version.

    Returns the new version number.
    """
    print(f"Registering existing models as new version of '{name}'...")
    version = bootstrap_register_existing(name=name)
    print(f"  ✓ Registered as version {version}")

    print(f"Transitioning v{version} → Staging...")
    transition_stage(name, version, "Staging", archive_existing=True)
    print(f"  ✓ Transitioned to Staging")
    return version


def cmd_promote(name: str) -> int | None:
    """Promote the latest Staging version to Production.

    Returns the promoted version number, or None if nothing to promote.
    """
    staging_version = get_latest_version(name, stage="Staging")
    if staging_version is None:
        print(
            f"  ✗ No Staging version found for '{name}'. "
            f"Run with --bootstrap first.",
            file=sys.stderr,
        )
        return None

    print(f"Promoting {name} v{staging_version}: Staging → Production...")
    transition_stage(name, staging_version, "Production", archive_existing=True)
    print(f"  ✓ Promoted v{staging_version} to Production")
    return staging_version


def cmd_list(name: str) -> None:
    """Print the current state of the registry for `name`."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()

    try:
        rm = client.get_registered_model(name)
    except mlflow.exceptions.RestException:
        print(f"  ℹ '{name}' is not registered. Run with --bootstrap to create it.")
        return

    print(f"Registered model: {rm.name}")
    print(f"  Created: {rm.creation_timestamp}")
    print(f"  Description: {rm.description or '(none)'}")
    print()

    all_versions = client.search_model_versions(f"name='{name}'")
    if not all_versions:
        print("  (no versions yet)")
        return

    print(f"  {'Version':<8} {'Stage':<12} {'Status':<10} {'Run ID':<32}")
    print(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*32}")
    for v in sorted(all_versions, key=lambda x: int(x.version)):
        print(
            f"  {v.version:<8} {v.current_stage:<12} "
            f"{v.status:<10} {v.run_id[:30]:<32}"
        )


def print_next_steps(model_name: str) -> None:
    """Print helpful UI/curl commands the user can copy-paste."""
    print()
    print("=" * 70)
    print("Next steps:")
    print()
    print("  1. View the registry in the MLflow UI:")
    print()
    print("     mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000")
    print()
    print("     Then open: http://localhost:5000/#/models")
    print()
    print(f"  2. Test loading the model from registry:")
    print()
    print(f"     python -c \"")
    print(f"     from src.registry import load_pyfunc_model")
    print(f"     m = load_pyfunc_model('{model_name}', stage='Staging')")
    print(f"     print('Loaded:', type(m).__name__)\"")
    print()
    print(f"  3. Run the API in registry mode (instead of file mode):")
    print()
    print(f"     $env:MODEL_SOURCE='registry:Production'   # PowerShell")
    print(f"     uv run uvicorn api.main:app --port 8000")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Register existing pickle files in models/ as a new version (→ Staging)",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote the latest Staging version to Production",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print current registry state",
    )
    parser.add_argument(
        "--name",
        default=MODEL_NAME,
        help=f"Registered model name (default: {MODEL_NAME})",
    )
    args = parser.parse_args()

    if not (args.bootstrap or args.promote or args.list):
        parser.print_help()
        sys.exit(1)

    if args.bootstrap:
        cmd_bootstrap(args.name)

    if args.promote:
        cmd_promote(args.name)

    if args.list:
        cmd_list(args.name)

    if args.bootstrap or args.promote:
        print_next_steps(args.name)


if __name__ == "__main__":
    main()
