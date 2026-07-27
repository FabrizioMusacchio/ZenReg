"""
Create synthetic example datasets for ZenReg.

The script writes canonical TZCYX OME-TIFF benchmark stacks into
``example_data/synthetic_data``:
- 2D+t with global XY time shifts;
- 3D with per-slice XY shifts;
- 3D+t with global XY time shifts;
- 3D+t with intra-stack-only XY slice shifts;
- 3D+t with global ZYX time shifts.
- 2D+t with global in-plane rotation.

The implementation uses ``zenreg.synthetic`` so the same data-generation code
can later be extended as part of the package itself.

Author: Fabrizio Musacchio
Date: June 2026
"""
# %% IMPORTS
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zenreg.synthetic import write_example_dataset


# %% MAIN
def main() -> None:
    """Generate the default ZenReg synthetic example datasets."""

    output_dir = PROJECT_ROOT / "example_data" / "synthetic_data"
    paths = write_example_dataset(output_dir)
    print("Wrote ZenReg synthetic example data:")
    for label, path in paths.items():
        print(f"  {label}: {path}")

# %% MAIN EXECUTION
if __name__ == "__main__":
    main()
# %%  END
