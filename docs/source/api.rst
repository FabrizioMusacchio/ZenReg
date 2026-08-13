API Reference
=============

This page documents the public ZenReg functions used by the tutorials. The
main entry point for registration workflows is :func:`register_stack`.

.. currentmodule:: zenreg

Core Workflow
-------------

.. autofunction:: load_stack
.. autofunction:: register_stack
.. autofunction:: save_stack
.. autofunction:: cleanup_omio_cache
.. autofunction:: create_empty_stack
.. autofunction:: crop_stack

Registration Backends
---------------------

.. autofunction:: register_stack_normcorre
.. autofunction:: register_stack_rigid_3d
.. autofunction:: correct_intra_stack_z_drift

Batch And Compute Helpers
-------------------------

.. autofunction:: discover_bids_like_batch_images
.. autofunction:: register_bids_like_batch
.. autofunction:: batch_create_thorlabs_raw_yaml_templates
.. autoclass:: BatchImageRecord
   :members:
.. autoclass:: BatchProcessedRecord
   :members:
.. autoclass:: BatchSkippedRecord
   :members:
.. autoclass:: BatchRegistrationResult
   :members:
.. autoclass:: BatchRawYamlTemplateRecord
   :members:
.. autoclass:: BatchRawYamlTemplateResult
   :members:
.. autofunction:: available_cpu_count
.. autofunction:: print_available_compute

Tutorial And Plotting Helpers
-----------------------------

.. autofunction:: plot_normcorre_patch_overlay
.. autofunction:: open_in_napari
.. autofunction:: show_before_after
.. autofunction:: show_projection
.. autofunction:: show_timepoints
.. autofunction:: show_slices
.. autofunction:: write_registration_summary_plot
.. autofunction:: print_shift_comparison
.. autofunction:: print_rigid_comparison

Synthetic Data
--------------

.. currentmodule:: zenreg.synthetic

.. autofunction:: write_example_dataset
.. autofunction:: write_batch_example_project
.. autofunction:: create_2d_motion_distorted_stack
.. autofunction:: create_2d_variable_snr_motion_distorted_stack
.. autofunction:: create_3d_time_xy_motion_distorted_stack
.. autofunction:: create_3d_time_zyx_motion_distorted_stack
.. autofunction:: create_3d_time_rigid_motion_distorted_stack
