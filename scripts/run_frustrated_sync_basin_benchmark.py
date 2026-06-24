# -*- coding: utf-8 -*-

"""Compatibility wrapper for the frustrated sync basin benchmark."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from frustrated_sync_dynamics.run_basin_benchmark import main  # noqa: E402


if __name__ == "__main__":
    main()
