Batch processing
================

ZenReg can be used in simple loops, but it also provides a true 
`BIDS <https://bids.neuroimaging.io/index.html>`_-like batch
processor that discovers image files, loads them with OMIO, registers them,
saves the results, and writes batch-level error reports.

BIDS-like file tree
-------------------

The batch processor assumes a project tree with subject folders and one or more
experiment or tag-folder levels below each subject. Folder names do not need to
match a strict standard; ZenReg searches for configurable name tokens.

.. code-block:: text

   project_root
   ├─ <sub*>
   │  ├─ <exp*>
   │  │  ├─ image_01.tif / image_01.ome.tif / image_01.lsm / image_01.czi / image_01.raw
   │  ├─ <exp*>
   │  │  ├─ image_01.tif / image_01.ome.tif / image_01.lsm / image_01.czi / image_01.raw
   │  │  ├─ image_02.tif / image_02.ome.tif / image_02.lsm / image_02.czi / image_02.raw
   │  │  └─ ...
   │  ├─ <exp*>
   │  │  ├─ <tagfolder*>01
   │  │  │  ├─ image_01.tif / image_01.czi / image_01.raw / ...
   │  │  │  └─ ...
   │  │  ├─ <tagfolder*>02
   │  │  │  ├─ image_02.tif / image_02.czi / image_02.raw / ...
   │  │  │  └─ ...
   │  │  └─ ...
   │  └─ ...
   └─ <sub*>
      └─ ...

For example, subject folders may start with ``ID`` and experiment folders may
contain tags such as ``TP000``, ``TP001``, ``DA000_FOV``, or ``TL_000``.
Please refer to the `BIDS <https://bids.neuroimaging.io/getting_started/folders_and_files/folders.html>`_ 
documentation for a full list of recommended BIDS folder naming conventions.

All examples shown here can be found in a ready-to-run script in
ZenReg's repository:

.. code-block:: bash

   user_scripts/register_batch_bids_like_synthetic.py

Create a synthetic batch project
--------------------------------

ZenReg comes with a built-in synthetic project generator that 
creates a small BIDS-like folder tree with two subjects, two 
experiments, and a few synthetic images:

.. code-block:: python

   from pathlib import Path
   from zenreg.synthetic import write_batch_example_project

   project_root = Path("example_data") / "synthetic_batch_project"

   write_batch_example_project(
       project_root,
       subject_ids=("ID000001", "ID000002"),
       experiment_tags=("TP000", "TP001"),
       overwrite=True)

This creates a BIDS-like folder, which looks as follows:

.. code-block:: text

   example_data
   └─ synthetic_batch_project
      ├─ ID000001
      │  ├─ TP000
      │  │  ├─ image_01.tif
      │  │  └─ image_02.tif
      │  └─ TP001
      │     ├─ image_01.tif
      │     └─ image_02.tif
      └─ ID000002
         ├─ TP000
         │  ├─ image_01.tif
         │  └─ image_02.tif
         └─ TP001
            ├─ image_01.tif
            └─ image_02.tif

Here, ``example_data/synthetic_batch_project`` is the project root, ``ID000001`` and
``ID000002`` are subject folders, and ``TP000`` and ``TP001`` are experiment folders, 
which contain synthetic TIFF images. 

The synthetic project is useful for testing custom loops or the built-in batch
processor before mapping the same logic to real data.

Writing your own custom loop
-----------------------------

For custom project layouts, an explicit Python loop can be the clearest solution. 
The following example discovers subject and experiment folders, and processes the 
images within them:

.. code-block:: python

   from pathlib import Path
   from zenreg import cleanup_omio_cache, load_stack, register_stack, save_stack

   project_root = Path("example_data") / "synthetic_batch_project"
   selected_subjects = None  # or ("ID000001", "ID000002")
   selected_experiments = ("TP000", "TP001")
   image_patterns = ("*.ome.tif", "*.tif", "*.tiff", "*.czi", "*.lsm", "*.raw")
   cache_dir = project_root / "omio_memmap_cache"

   if selected_subjects is None:
       subject_dirs = sorted(
           path for path in project_root.iterdir()
           if path.is_dir() and path.name.startswith("ID"))
   else:
       subject_dirs = [project_root / subject_id for subject_id in selected_subjects]

   for subject_dir in subject_dirs:
       experiment_dirs = sorted(
           path for path in subject_dir.iterdir()
           if path.is_dir() and path.name in selected_experiments)

       for experiment_dir in experiment_dirs:
           image_files = []
           for pattern in image_patterns:
               image_files.extend(experiment_dir.glob(pattern))

           for image_path in sorted(set(image_files)):
               output_dir = experiment_dir / "zenreg_output"
               output_path = output_dir / f"{image_path.stem}_zenreg_registered.ome.tif"

               cleanup_omio_cache(cache_dir, full_cleanup=True)

               image, metadata = load_stack(
                   image_path,
                   return_metadata  = True,
                   use_memmap       = True,
                   memmap_folder    = cache_dir,
                   memmap_reuse     = True,
                   on_error         = "raise")

               registered, details = register_stack(
                   image,
                   registration_channel             = 0,
                   method                           = "phase_cross_correlation",
                   time_registration_mode           = "projection",
                   time_reference_mode              = "template",
                   registration_template_time_range = "all",
                   projection_method                = "max",
                   zreg                             = False,
                   zero_clip                        = True,
                   max_xy_shifts                    = (8, 8),
                   output_use_memmap                = True,
                   output_memmap_folder             = cache_dir,
                   n_jobs                           = -1,
                   verbose                          = True,
                   return_shifts                    = True,
                   return_details                   = True)

               save_stack(
                   output_path,
                   registered,
                   metadata             = metadata,
                   registration_details = details,
                   overwrite            = True)

               cleanup_omio_cache(cache_dir, full_cleanup=True)

Using ZenReg's built-in batch processor
----------------------------------------

ZenReg comes with a built-in batch processor called ``register_bids_like_batch``
that can be configured to discover and process image files in a project directory, 
which follows a BIDS-like folder structure. The entire functionality of ZenReg's core
functions (``load_stack``, ``register_stack``, and ``save_stack``) is available through 
this single function, which also handles error reporting and logging:

.. code-block:: python

   from pathlib import Path
   from zenreg import register_bids_like_batch

   project_root = Path("example_data") / "synthetic_batch_project"

   result = register_bids_like_batch(
       project_root,
       subject_ids          = None, # set to None to discover all folders starting with subject_prefix
       subject_prefix       = "ID", # the prefix for subject folders, e.g. "ID" for "ID20810"
       tag_folder_levels    = (("TP000", "TP001"),), # this is a tuple of tuples, one tuple per folder level below each subject
       image_patterns       = ("*.ome.tif", "*.tif", "*.czi", "*.lsm", "*.raw"),
       output_folder_name   = "zenreg_output",
       skip_registered      = True, # switch controlling file skipping if the expected registered output already exists
       use_memmap           = True,
       memmap_folder_name   = "omio_memmap_cache",
       memmap_reuse         = True,
       cleanup_cache_before_load    = False,
       cleanup_cache_after_save     = True,
       load_kwargs={"on_error": "return_none",
                    "verbose":  False,},
       register_kwargs={
           "registration_channel":              0,
           "method":                            "phase_cross_correlation",
           "time_registration_mode":            "projection",
           "time_reference_mode":               "template",
           "registration_template_time_range":  "all",
           "projection_method":                 "max",
           "zreg":                              False,
           "zero_clip":                         True,
           "max_xy_shifts":                     (8, 8),
           "n_jobs":                            -1,
           "verbose":                           True,
           "return_shifts":                     True,
           "return_details":                    True,},
       save_kwargs={
           "compression_level": 3,
           "overwrite":         True,
           "verbose":           False},
       write_error_reports      = True,
       continue_on_error        = True,
       verbose                  = True)

   print(f"Processed files: {len(result.processed)}")
   print(f"Skipped files:   {len(result.skipped)}")
   print(f"Error report:    {result.root_error_report_path}")

Nested tag folders are expressed by adding levels to ``tag_folder_levels``. This
example searches for folders containing ``DC000_FOV`` or ``DA000_FOV`` and then
for a nested ``TL_000`` folder:

.. code-block:: python

   result = register_bids_like_batch(
       project_root,
       subject_ids          = ("ID20810", "ID20867"),
       subject_prefix       = "ID",
       tag_folder_levels    =(
                                ("DC000_FOV", "DA000_FOV"),
                                ("TL_000",)
                             ),
       image_patterns       = ("*.raw", "*.ome.tif", "*.tif", "*.czi", "*.lsm"),
       use_memmap           = True,
       load_kwargs          = {"on_error": "return_none"},
       register_kwargs={
           "registration_channel":              1,
           "method":                            "phase_cross_correlation",
           "time_registration_mode":            "projection",
           "registration_template_time_range":  (0, 500),
           "max_xy_shifts":                     (110, 110),
           "n_jobs":                            20,
           "return_details":                    True,},)

Options introduced here:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Argument
     - Meaning
   * - ``subject_ids``
     - Explicit subject folders to process. Use ``None`` to discover all
       subject folders whose names start with ``subject_prefix``.
   * - ``subject_prefix``
     - Folder-name prefix for automatic subject discovery. Default: ``"ID"``.
   * - ``tag_folder_levels``
     - One tuple/list per folder level below each subject. Tokens are matched by
       containment, so ``("DC000_FOV",)`` matches folders such as
       ``DC000_FOV1``.
   * - ``image_patterns``
     - Glob patterns for image discovery in the final tag-folder level.
   * - ``skip_registered``
     - Skip files whose expected registered output already exists. Default:
       ``True``.
   * - ``load_kwargs``
     - Keyword arguments forwarded to ``load_stack``. In batch mode,
       ``on_error="return_none"`` lets ZenReg skip unreadable files instead of
       aborting the full run.
   * - ``register_kwargs``
     - Keyword arguments forwarded to ``register_stack``.
   * - ``save_kwargs``
     - Keyword arguments forwarded to ``save_stack``.
   * - ``use_memmap``
     - Enables OMIO disk-backed loading and disk-backed registered-output
       caches unless these settings are explicitly overridden.
   * - ``write_error_reports``
     - Writes short per-folder reports and a root-level, copy-pasteable Python
       dictionary of skipped files. Default: ``True``.
   * - ``continue_on_error``
     - Continue with the next image after load, registration, or save errors.
       Default: ``True``.

All other keyword arguments are the same as for the individual functions, and are 
forwarded to them. Please refer to the corresponding function documentation and
tutorials for details.


Batch error reports
-------------------

If an image cannot be loaded, registered, or saved, ZenReg records the failure
and continues when ``continue_on_error=True``. I.e, the batch processor does not 
abort on the first error, but instead writes a root-level error report and continues 
with the next image. The root-level error report is a Python dictionary stored in a 
text file. For Thorlabs RAW files with missing or inconsistent metadata, the root 
error report also contains editable metadata defaults:

.. code-block:: python

   ZENREG_BATCH_SKIPPED_RAW_FILES = {
       "/path/to/Image_001_001.raw": {
           "reason": "...",
           "stage": "load",
           "subject_id": "ID20810",
           "tag_folders": ("DC000_FOV1", "TL_000"),
           "reported_at": "2026-08-10_12-00-00",
           "template_metadata": {
               "T": 1,
               "Z": 1,
               "C": 1,
               "Y": 1,
               "X": 1,
               "bits": 16,
               "pixelunit": "micron",
               "physicalsize_xyz": (0.5, 0.5, 1.0),
               "time_increment": 1.0,
               "time_increment_unit": "seconds",
           },
       },
   }

The user can edit these metadata values centrally and then create 
`OMIO YAML templates <https://omio.readthedocs.io/en/latest/usage_file_format_supported.html#reading-thorlabs-raw-files>`_
for the skipped RAW files. Also see the next section for a helper function 
that automates this process.

Creating Thorlabs RAW YAML templates
------------------------------------

For Thorlabs RAW files with missing or inconsistent XML metadata, OMIO can use
`YAML sidecar files <https://omio.readthedocs.io/en/latest/usage_file_format_supported.html#reading-thorlabs-raw-files>`_ 
as explicit metadata bypass files. ZenReg provides a helper
that reads the root batch error report, extracts skipped RAW paths and their
``template_metadata`` blocks, and calls OMIO's YAML-template creator for each
RAW file listed in the report:

.. code-block:: python

   from pathlib import Path
   from zenreg import batch_create_thorlabs_raw_yaml_templates

   result = batch_create_thorlabs_raw_yaml_templates(
       project_root,
       report_name        = "zenreg_batch_error_report_2026-08-10_12-00-00.txt",
       overwrite_existing = False,
       verbose            = True)

   print(f"Created YAML templates: {len(result.created)}")
   print(f"Skipped RAW files:      {len(result.skipped)}")

Set ``report_name=None`` to use the latest
``zenreg_batch_error_report_*.txt`` in ``project_root``. The function does not
scan the project tree again. It deliberately trusts the report and creates YAML
templates only for the RAW paths listed there.

.. tip::

    The error report created by ZenReg contains a valid Python dictionary. The
    intended workflow is to edit the metadata values centrally in that report
    and then let ZenReg create YAML templates for all listed RAW files in one
    go.

Options introduced here:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Argument
     - Meaning
   * - ``report_name``
     - Root-level ZenReg batch error report. Use ``None`` to choose the latest
       matching report in ``project_root``.
   * - ``raw_template_metadata``
     - Fallback metadata for older reports without per-file
       ``template_metadata`` blocks. Current reports should usually be edited
       directly.
   * - ``overwrite_existing``
     - If ``False`` (default), existing YAML/YML sidecars are kept and the RAW
       file is skipped.

ZenReg's GitHub repository also contains a ready-to-run script that demonstrates the 
Thorlabs RAW metadata repair workflows illustrated above:

.. code-block:: bash

   user_scripts/OMIO_yaml_template_creator.py 
