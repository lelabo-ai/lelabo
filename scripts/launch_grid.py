"""Backward-compatible launcher entrypoint.

Preferred location: experiments/launchers/launch_grid.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.launchers.launch_grid import main


if __name__ == "__main__":
    main()
