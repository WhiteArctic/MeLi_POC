from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_moderation_poc.cli import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "diagnose-url", *sys.argv[1:]]
    main()
