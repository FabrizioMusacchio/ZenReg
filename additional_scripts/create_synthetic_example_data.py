"""
Create synthetic example datasets for ZenReg.

The script writes two canonical TZCYX OME-TIFF stacks into ``example_data/synthetic_data``:
- a 2D time-lapse stack with global time-wise motion artifacts;
- a 3D time-lapse stack with global time-wise motion and intra-stack Z drift.

The implementation uses ``zenreg.synthetic`` so the same data-generation code
can later be extended as part of the package itself.

Author: Fabrizio Musacchio
Date: June 2026
"""
# ruff: noqa: I001

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zenreg.synthetic import write_example_dataset


def main() -> None:
    """Generate the default ZenReg synthetic example datasets."""

    output_dir = PROJECT_ROOT / "example_data" / "synthetic_data"
    paths = write_example_dataset(output_dir)
    print("Wrote ZenReg synthetic example data:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
