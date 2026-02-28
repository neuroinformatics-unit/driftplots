"""Example: run the hover viewer.

Usage:
  python example_run_hover_viewer.py /path/to/kilosort/output --ks_version kilosort4 --decimate 20
"""

from __future__ import annotations
import argparse
from hover_viewer_pyqtgraph import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sorter_output", type=str)
    ap.add_argument("--ks_version", type=str, default="kilosort4", help="kilosort4 or kilosort1_3")
    args = ap.parse_args()

    run(args.sorter_output, ks_version="kilosort4")


if __name__ == "__main__":
    main()
