"""Orchestrates the Bronze -> Silver -> Gold medallion pipeline end to end.

Usage: python run_pipeline.py
"""
import time

from medallion.bronze.ingest import ingest
from medallion.gold.features import run as build_features
# from medallion.gold.snapshots import run as build_snapshots
from medallion.silver.clean import clean


def main():
    t0 = time.time()
    print("=== Bronze: ingest ===")
    ingest()
    print(f"\n=== Silver: clean ===  ({time.time() - t0:.1f}s elapsed)")
    clean()
    print(f"\n=== Gold: features ===  ({time.time() - t0:.1f}s elapsed)")
    build_features()
    # print(f"\n=== Gold: snapshots ===  ({time.time() - t0:.1f}s elapsed)")
    # build_snapshots()
    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
