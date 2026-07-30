Batch Processing
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
