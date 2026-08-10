Batch Processing
================

ZenReg can be used in simple loops, but it also provides a true BIDS-like batch
processor that discovers image files, loads them with OMIO, registers them,
saves the results, and writes batch-level error reports.

BIDS-like File Tree
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
contain tags such as ``TP000``, ``DC000_FOV``, ``DA000_FOV``, or ``TL_000``.

Create A Synthetic Batch Project
--------------------------------

The repository includes a ready-to-run script:

.. code-block:: bash

   python user_scripts/register_batch_bids_like_synthetic.py

The script first creates a small synthetic project:

.. code-block:: python

   from pathlib import Path
   from zenreg.synthetic import write_batch_example_project

   project_root = Path("example_data") / "synthetic_batch_project"

   write_batch_example_project(
       project_root,
       subject_ids=("ID000001", "ID000002"),
       experiment_tags=("TP000", "TP001"),
       overwrite=True,
   )

The synthetic project is useful for testing custom loops or the built-in batch
processor before mapping the same logic to real data.

Writing Your Own Loop
---------------------

For unusual project layouts, an explicit Python loop can still be the clearest
choice. The registration calls remain ordinary ZenReg calls:

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
           if path.is_dir() and path.name.startswith("ID")
       )
   else:
       subject_dirs = [project_root / subject_id for subject_id in selected_subjects]

   for subject_dir in subject_dirs:
       experiment_dirs = sorted(
           path for path in subject_dir.iterdir()
           if path.is_dir() and path.name in selected_experiments
       )

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
                   return_metadata=True,
                   use_memmap=True,
                   memmap_folder=cache_dir,
                   memmap_reuse=True,
                   on_error="raise",
               )

               registered, details = register_stack(
                   image,
                   registration_channel=0,
                   method="phase_cross_correlation",
                   time_registration_mode="projection",
                   time_reference_mode="template",
                   registration_template_time_range="all",
                   projection_method="max",
                   zreg=False,
                   zero_clip=True,
                   max_xy_shifts=(8, 8),
                   output_use_memmap=True,
                   output_memmap_folder=cache_dir,
                   n_jobs=-1,
                   verbose=True,
                   return_shifts=True,
                   return_details=True,
               )

               save_stack(
                   output_path,
                   registered,
                   metadata=metadata,
                   registration_details=details,
                   overwrite=True,
               )

               cleanup_omio_cache(cache_dir, full_cleanup=True)

Using The ZenReg Batch Processor
--------------------------------

For recurring analyses, ``register_bids_like_batch`` wraps discovery, loading,
registration, saving, cache handling, and error reporting in one call.

.. code-block:: python

   from pathlib import Path
   from zenreg import register_bids_like_batch

   project_root = Path("example_data") / "synthetic_batch_project"

   result = register_bids_like_batch(
       project_root,
       subject_ids=None,                 # None discovers all folders starting with subject_prefix
       subject_prefix="ID",
       tag_folder_levels=(("TP000", "TP001"),),
       image_patterns=("*.ome.tif", "*.tif", "*.czi", "*.lsm", "*.raw"),
       output_folder_name="zenreg_output",
       skip_registered=True,
       use_memmap=True,
       memmap_folder_name="omio_memmap_cache",
       memmap_reuse=True,
       cleanup_cache_before_load=False,
       cleanup_cache_after_save=True,
       load_kwargs={
           "on_error": "return_none",
           "verbose": False,
       },
       register_kwargs={
           "registration_channel": 0,
           "method": "phase_cross_correlation",
           "time_registration_mode": "projection",
           "time_reference_mode": "template",
           "registration_template_time_range": "all",
           "projection_method": "max",
           "zreg": False,
           "zero_clip": True,
           "max_xy_shifts": (8, 8),
           "n_jobs": -1,
           "verbose": True,
           "return_shifts": True,
           "return_details": True,
       },
       save_kwargs={
           "compression_level": 3,
           "overwrite": True,
           "verbose": False,
       },
       write_error_reports=True,
       continue_on_error=True,
       verbose=True,
   )

   print(f"Processed files: {len(result.processed)}")
   print(f"Skipped files:   {len(result.skipped)}")
   print(f"Error report:    {result.root_error_report_path}")

Nested tag folders are expressed by adding levels to ``tag_folder_levels``. This
example searches for folders containing ``DC000_FOV`` or ``DA000_FOV`` and then
for a nested ``TL_000`` folder:

.. code-block:: python

   result = register_bids_like_batch(
       project_root,
       subject_ids=("ID20810", "ID20867"),
       subject_prefix="ID",
       tag_folder_levels=(
           ("DC000_FOV", "DA000_FOV"),
           ("TL_000",),
       ),
       image_patterns=("*.raw", "*.ome.tif", "*.tif", "*.czi", "*.lsm"),
       use_memmap=True,
       load_kwargs={"on_error": "return_none"},
       register_kwargs={
           "registration_channel": 1,
           "method": "phase_cross_correlation",
           "time_registration_mode": "projection",
           "registration_template_time_range": (0, 500),
           "max_xy_shifts": (110, 110),
           "n_jobs": 20,
           "return_details": True,
       },
   )

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

Batch Error Reports
-------------------

If an image cannot be loaded, registered, or saved, ZenReg records the failure
and continues when ``continue_on_error=True``. For Thorlabs RAW files with
missing or inconsistent metadata, the root error report also contains editable
metadata defaults:

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

The user can edit these metadata values centrally and then create OMIO YAML
templates for the skipped RAW files.

Ready-To-Adapt Script
---------------------

The complete synthetic example is available as
``user_scripts/register_batch_bids_like_synthetic.py`` in the repository. It
contains both a simple explicit loop and a ``register_bids_like_batch`` example
that can be copied into a project-specific batch script.
