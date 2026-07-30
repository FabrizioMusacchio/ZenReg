ZenReg Documentation
====================

.. image:: _static/ZenReg_logo_wide.jpg
   :alt: ZenReg logo
   :align: center
   :width: 100%

|

.. image:: https://badgen.net/badge/icon/GitHub%20repository?icon=github&label
   :target: https://github.com/FabrizioMusacchio/ZenReg/
   :alt: GitHub repository

.. image:: https://img.shields.io/badge/License-GPL%20v3-green.svg
   :target: https://github.com/FabrizioMusacchio/ZenReg
   :alt: GPLv3 License

.. image:: https://readthedocs.org/projects/zenreg/badge/?version=latest
   :target: https://zenreg.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

ZenReg is a Python package for fast, memory-efficient, and modular microscopy
image registration. 

ZenReg's philosophy is to make common microscopy registration tasks available
through one convenient wrapper while keeping the internals modular enough for
new backends and contributions. The package is therefore designed around one main workflow
and a set of modular backends that can be used:

.. code-block:: python

   from zenreg import load_stack, register_stack, save_stack

   image, metadata = load_stack("image.ome.tif", return_metadata=True)

   registered, details = register_stack(
       image,
       registration_channel=0,
       method="phase_cross_correlation",
       return_shifts=True,
       return_details=True)

   save_stack("image_registered.ome.tif", registered, metadata=metadata, registration_details=details)

At the same time, the package is built for reproducible outputs:
Registered OME-TIFFs can be saved together with shift tables, settings YAML
files, and summary plots. This enables users to easily share their results and 
reproduce them later.

.. toctree::
   :maxdepth: 3
   :caption: Contents

   overview
   installation
   usage
   api
   changelog
   contributing
