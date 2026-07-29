"""Orchestrates the Bronze -> Silver -> Gold medallion pipeline end to end.

Usage: python medallion/run_pipeline.py  (or `python -m medallion.run_pipeline`)

Stops at Gold's training_features.parquet + item_coclick_snapshot.pkl -
the product/user scoring snapshots (gold/snapshots.py) aren't run here,
since the modeling notebooks already build the equivalent snapshots
inline when they need to score.
"""
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from medallion.bronze.ingest import ingest
from medallion.gold.features import run as build_features
from medallion.silver.clean import clean


def main():
    t0 = time.time()
    print("=== Bronze: ingest ===")
    ingest()
    print(f"\n=== Silver: clean ===  ({time.time() - t0:.1f}s elapsed)")
    clean()
    print(f"\n=== Gold: features ===  ({time.time() - t0:.1f}s elapsed)")
    build_features()
    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
