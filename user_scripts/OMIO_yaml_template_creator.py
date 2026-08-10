"""
Create OMIO Thorlabs RAW YAML templates from a ZenReg batch error report.

Run this script after a ZenReg batch run has produced a root-level
``zenreg_batch_error_report_<date>.txt``. Edit the per-file
``template_metadata`` blocks in that report first, then run this script to
create OMIO YAML bypass files next to the affected RAW images.

Activate the same conda environment that contains ZenReg and OMIO v0.2.8 or
newer.

Author: Fabrizio Musacchio
Date: August 2026
"""
# %% IMPORTS
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zenreg import create_thorlabs_raw_yaml_templates_from_batch_report
# %% BATCH-LIKE PATH SETTINGS
BATCH_ROOT = PROJECT_ROOT / "example_data" / "synthetic_batch_project"

# Use either explicit IDs or None to discover all folders starting with
# SUBJECT_PREFIX.
BATCH_IDS = ("ID000001", "ID000002")
SUBJECT_PREFIX = "ID"

# Same tag-folder logic as register_bids_like_batch:
# tokens are matched by containment, e.g. "DC000_FOV" matches "DC000_FOV1".
BATCH_TAG_FOLDER_LEVELS = (
    ("TP000", "TP001"),)

# Name of the root-level report written by register_bids_like_batch.
# Set to None to use the latest "zenreg_batch_error_report_*.txt" in BATCH_ROOT.
ZENREG_BATCH_ERROR_REPORT_NAME = "zenreg_batch_error_report_YYYY-MM-DD_HH-MM-SS.txt"

# Existing YAML/YML files are skipped by default.
OVERWRITE_EXISTING_YML_FILES = False

# These defaults are only used for legacy reports without per-file
# template_metadata blocks. For current ZenReg reports, edit the metadata inside
# the report itself.
RAW_TEMPLATE_METADATA_FALLBACK = dict(
    T=1,
    Z=1,
    C=1,
    Y=1,
    X=1,
    bits=16,
    pixelunit="micron",
    physicalsize_xyz=(0.5, 0.5, 1.0),
    time_increment=1.0,
    time_increment_unit="seconds",)
# %% CREATE YAML TEMPLATES FROM ZENREG BATCH ERROR REPORT
result = create_thorlabs_raw_yaml_templates_from_batch_report(
    BATCH_ROOT,
    report_name             = ZENREG_BATCH_ERROR_REPORT_NAME,
    subject_ids             = BATCH_IDS,
    subject_prefix          = SUBJECT_PREFIX, # this only affects when subject_ids=None or commented out
    tag_folder_levels       = BATCH_TAG_FOLDER_LEVELS,
    image_patterns          = ("*.raw",),
    exclude_name_contains   = ("ROIMask.raw",),
    restrict_to_discovered  = True,
    raw_template_metadata   = RAW_TEMPLATE_METADATA_FALLBACK,
    overwrite_existing      = OVERWRITE_EXISTING_YML_FILES,
    verbose                 = True,)

print(f"Report:  {result.report_path}")
print(f"Created: {len(result.created)}")
print(f"Skipped: {len(result.skipped)}")
for record in result.skipped:
    print(f"  skipped [{record.status}]: {record.raw_path} ({record.reason})")
# %% END
