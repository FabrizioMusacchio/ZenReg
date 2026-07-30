Batch processing
================

ZenReg does not require a special batch runner. A robust batch workflow is
usually a small loop around:

.. code-block:: text

   load_stack -> register_stack -> save_stack

The helper ``iter_bids_like_image_files`` can discover images in a simple
BIDS-like project structure.

Example project layout
----------------------

.. code-block:: text

   project_root
   ├─ ID000001
   │  ├─ TP000
   │  │  ├─ image_01.ome.tif
   │  ├─ TP001
   │  │  ├─ image_01.ome.tif
   │  │  ├─ image_02.ome.tif
   │  │  └─ ...
   │  └─ ...
   └─ ID000002
      └─ ...

The folder names are intentionally simple: subjects start with ``ID`` and
experiments/time points start with ``TP``.

Create the synthetic batch project
----------------------------------

The repository includes an interactive script:

.. code-block:: bash

   python user_scripts/register_batch_bids_like_synthetic.py

The script first creates a small synthetic project in
``example_data/synthetic_batch_project``:

.. code-block:: python

   from pathlib import Path
   from zenreg.synthetic import write_batch_example_project

   project_root = Path("example_data") / "synthetic_batch_project"
   write_batch_example_project(
       project_root,
       subject_ids=("ID000001", "ID000002"),
       experiment_tags=("TP000", "TP001"),
   )

Discover files
--------------

Process all subjects whose folder name starts with ``ID``:

.. code-block:: python

   from zenreg import iter_bids_like_image_files

   records = iter_bids_like_image_files(
       project_root,
       subject_ids=None,
       experiment_tags=("TP000", "TP001"),
       subject_prefix="ID",
       experiment_prefix="TP",
   )

Or restrict the run to selected subjects:

.. code-block:: python

   records = iter_bids_like_image_files(
       project_root,
       subject_ids=("ID000001",),
       experiment_tags=("TP000",),
   )

Options introduced here
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Argument
     - Meaning
     - Default behavior
   * - ``project_root``
     - Root folder that contains subject folders.
     - Required.
   * - ``subject_ids``
     - Explicit subject folders to process. Use ``None`` to discover all
       matching subjects.
     - ``None``.
   * - ``experiment_tags``
     - Experiment folders to include, for example ``("TP000", "TP001")``.
     - ``None`` discovers matching experiment folders.
   * - ``subject_prefix`` / ``experiment_prefix``
     - Folder-name prefixes used for automatic discovery.
     - ``"ID"`` and ``"TP"``.

Batch loop
----------

The actual registration loop is ordinary ZenReg code:

.. code-block:: python

   from zenreg import cleanup_omio_cache, load_stack, register_stack, save_stack

   cache_dir = project_root / "omio_memmap_cache"

   for record in records:
       output_dir = record.image_path.parent / "zenreg_registered"
       output_path = output_dir / record.image_path.name

       cleanup_omio_cache(cache_dir, full_cleanup=True)
       image, metadata = load_stack(
           record.image_path,
           return_metadata=True,
           use_memmap=True,
           memmap_folder=cache_dir,
           memmap_reuse=True,
       )

       registered, details = register_stack(
           image,
           registration_channel=0,
           registration_stack=0,
           method="phase_cross_correlation",
           time_registration_mode="projection",
           projection_method="max",
           zreg=False,
           zero_clip=True,
           max_xy_shifts=(8, 8),
           output_use_memmap=True,
           output_memmap_folder=cache_dir,
           n_jobs=4,
           return_shifts=True,
           return_details=True,
       )

       save_stack(output_path, registered, metadata=metadata, registration_details=details)
       cleanup_omio_cache(cache_dir, full_cleanup=True)

Output location
---------------

The example writes registered files next to each input image:

.. code-block:: text

   ID000001/TP000/zenreg_registered/image_01.ome.tif

The report sidecars are written next to the registered image when
``registration_details`` is passed to ``save_stack``.

Writing your own loop
---------------------

The helper above is intentionally small. If your project structure differs, it
is often clearer to write the iteration logic directly and keep ZenReg calls
unchanged:

.. code-block:: python

   from pathlib import Path
   from zenreg import cleanup_omio_cache, load_stack, register_stack, save_stack

   project_root = Path("example_data") / "synthetic_batch_project"
   selected_subjects = None  # or ("ID000001", "ID000002")
   selected_experiments = ("TP000", "TP001")
   image_extensions = (".tif", ".tiff", ".ome.tif", ".ome.tiff", ".czi", ".lsm", ".raw")
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
           if path.is_dir()
           and path.name.startswith("TP")
           and path.name in selected_experiments
       )

       for experiment_dir in experiment_dirs:
           image_files = sorted(
               path for path in experiment_dir.iterdir()
               if path.is_file() and path.name.lower().endswith(image_extensions)
           )

           for image_path in image_files:
               output_path = experiment_dir / "zenreg_registered" / image_path.name

               cleanup_omio_cache(cache_dir, full_cleanup=True)
               image, metadata = load_stack(
                   image_path,
                   return_metadata=True,
                   use_memmap=True,
                   memmap_folder=cache_dir,
                   memmap_reuse=True,
               )

               registered, details = register_stack(
                   image,
                   registration_channel=0,
                   registration_stack=0,
                   method="phase_cross_correlation",
                   time_registration_mode="projection",
                   projection_method="max",
                   zreg=False,
                   max_xy_shifts=(8, 8),
                   zero_clip=True,
                   output_use_memmap=True,
                   output_memmap_folder=cache_dir,
                   n_jobs=-1,
                   return_shifts=True,
                   return_details=True,
               )

               save_stack(output_path, registered, metadata=metadata, registration_details=details)
               cleanup_omio_cache(cache_dir, full_cleanup=True)

This pattern is useful when folder names, inclusion rules, or output names are
project-specific.

New options in this block
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Name
     - Meaning
     - Typical value
   * - ``selected_subjects``
     - Set to ``None`` to process all discovered subjects, or provide an
       explicit tuple/list of subject IDs.
     - ``None`` or ``("ID000001",)``.
   * - ``selected_experiments``
     - Experiment folders to include in the manual loop.
     - ``("TP000", "TP001")``.
   * - ``image_extensions``
     - File extensions treated as image inputs.
     - TIFF/OME-TIFF/CZI/LSM/RAW suffixes.
   * - ``n_jobs=-1``
     - Use all available CPU workers for parallelizable registration work.
     - ``1`` unless explicitly changed.
