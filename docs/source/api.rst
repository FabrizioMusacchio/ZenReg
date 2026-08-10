API Reference
=============

This page documents the public ZenReg functions used by the tutorials. The
main entry point for registration workflows is :func:`zenreg.register_stack`.

Core workflow
-------------

.. autosummary::
   :toctree: generated

   zenreg.load_stack
   zenreg.register_stack
   zenreg.save_stack
   zenreg.cleanup_omio_cache
   zenreg.create_empty_stack
   zenreg.crop_stack

Registration backends
---------------------

.. autosummary::
   :toctree: generated

   zenreg.register_stack_normcorre
   zenreg.register_stack_rigid_3d
   zenreg.correct_intra_stack_z_drift

Batch and compute helpers
-------------------------

.. autosummary::
   :toctree: generated

   zenreg.discover_bids_like_batch_images
   zenreg.register_bids_like_batch
   zenreg.BatchImageRecord
   zenreg.BatchProcessedRecord
   zenreg.BatchSkippedRecord
   zenreg.BatchRegistrationResult
   zenreg.available_cpu_count
   zenreg.print_available_compute

Tutorial and plotting helpers
-----------------------------

.. autosummary::
   :toctree: generated

   zenreg.plot_normcorre_patch_overlay
   zenreg.open_in_napari
   zenreg.show_before_after
   zenreg.show_projection
   zenreg.show_timepoints
   zenreg.show_slices
   zenreg.write_registration_summary_plot
   zenreg.print_shift_comparison
   zenreg.print_rigid_comparison

Synthetic data
--------------

.. autosummary::
   :toctree: generated

   zenreg.synthetic.write_example_dataset
   zenreg.synthetic.write_batch_example_project
   zenreg.synthetic.create_2d_motion_distorted_stack
   zenreg.synthetic.create_3d_time_xy_motion_distorted_stack
   zenreg.synthetic.create_3d_time_zyx_motion_distorted_stack
   zenreg.synthetic.create_3d_time_rigid_motion_distorted_stack

Modules
-------

.. automodule:: zenreg
   :members:
   :undoc-members:
   :no-index:

.. automodule:: zenreg.batch
   :members:
   :undoc-members:
   :no-index:

.. automodule:: zenreg.io
   :members:
   :undoc-members:
   :no-index:

.. automodule:: zenreg.registration
   :members:
   :undoc-members:
   :no-index:

.. automodule:: zenreg.normcorre
   :members:
   :undoc-members:
   :no-index:

.. automodule:: zenreg.rigid3d
   :members:
   :undoc-members:
   :no-index:

.. automodule:: zenreg.synthetic
   :members:
   :undoc-members:
   :no-index:
