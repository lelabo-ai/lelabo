"""Backward-compatible plotting entrypoint.

Preferred location: tools/plot_sweep_table.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.plot_sweep_table import main


if __name__ == "__main__":
    main()
