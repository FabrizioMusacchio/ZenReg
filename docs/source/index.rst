ZenReg Documentation
====================

.. image:: https://badgen.net/badge/icon/GitHub%20repository?icon=github&label
   :target: https://github.com/FabrizioMusacchio/ZenReg/
   :alt: GitHub repository

.. image:: https://img.shields.io/badge/License-GPL%20v3-green.svg
   :target: https://github.com/FabrizioMusacchio/ZenReg
   :alt: GPLv3 License

.. image:: https://readthedocs.org/projects/zenreg/badge/?version=latest
   :target: https://zenreg.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

ZenReg is a Python package for modular microscopy image registration. It uses
`OMIO <https://github.com/FabrizioMusacchio/omio>`_ for microscopy I/O and
normalizes every input image to canonical ``TZCYX`` order before registration.

The package is designed around one main workflow:

.. code-block:: python

   from zenreg import load_stack, register_stack, save_stack

   image, metadata = load_stack("image.ome.tif", return_metadata=True)
   registered, details = register_stack(
       image,
       registration_channel=0,
       method="phase_cross_correlation",
       return_shifts=True,
       return_details=True,
   )
   save_stack("image_registered.ome.tif", registered, metadata=metadata, registration_details=details)

.. toctree::
   :maxdepth: 3
   :caption: Contents

   overview
   installation
   usage
   api
   whats_new
   contributing
