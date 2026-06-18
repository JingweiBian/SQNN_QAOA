# -*- coding: utf-8 -*-

"""Entry point for the long MaxCut-3 V14 research loop.

Keeping the arguments in Python avoids Windows background-launch quoting issues
with long command lines and paths containing spaces.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_maxcut3_v14_research_loop import main  # noqa: E402


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--output-dir",
        "outputs/maxcut3_v14_24h_research_live",
        "--baseline-dir",
        "outputs/maxcut3_v14_24h_research_live/baselines",
        "--device",
        "cuda",
        "--hours",
        "24",
        "--max-runs-per-cycle",
        "6",
        "--screen-rounds",
        "140",
        "--screen-epochs",
        "55",
        "--exploit-rounds",
        "260",
        "--exploit-epochs",
        "115",
        "--num-samples",
        "384",
        "--local-search-passes",
        "240",
        "--sample-local-search-passes",
        "120",
        "--skip-baseline",
        "--resume",
    ]
    main()
