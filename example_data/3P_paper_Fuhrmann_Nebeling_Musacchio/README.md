# Three-photon 3D example stack

This folder is reserved for the real 3P z-stack used in ZenReg's pseudo-time
3D registration example. The image file itself is not distributed with ZenReg.

## Download

ZenReg expects the file to be placed here:

```text
example_data/3P_paper_Fuhrmann_Nebeling_Musacchio/Supplementary_Video_4.tif
```

The stack is available from the original Dryad dataset:

<https://datadryad.org/dataset/doi:10.5061/dryad.tqjq2bw90>

For a smaller, single-file download, the same example stack is also included in
the OMIO example-data Zenodo archive:

<https://doi.org/10.5281/zenodo.18078231>

## Source, license, and citation

The original dataset is released under CC0 1.0 Universal:

<https://creativecommons.org/publicdomain/zero/1.0/>

Please cite the dataset if you use this file:

Fuhrmann, F., Nebeling, F. C., Musacchio, F., et al. (2025). *Data from:
Three-photon in vivo imaging of neurons and glia in the medial prefrontal
cortex with sub-cellular resolution* [Dataset]. Dryad.
<https://doi.org/10.5061/dryad.tqjq2bw90>

The dataset accompanies:

Fuhrmann, F., Nebeling, F. C., Musacchio, F., et al. (2025). Three-photon
in vivo imaging of neurons and glia in the medial prefrontal cortex with
sub-cellular resolution. *Communications Biology*, 8, 795.
<https://doi.org/10.1038/s42003-025-08079-8>

## File description

`Supplementary_Video_4.tif` contains an in vivo three-photon cortex-to-hippocampus
z-scan with 265 XY planes from the brain surface to 1325 micrometers below the
surface, acquired at 5 micrometer axial spacing with 1300 nm excitation in a
GFP.M::Cx3cr1-CreER::Rosa26_tdTomato transgenic mouse.

ZenReg uses this file as a real biological 3D volume. The accompanying
preprint example creates controlled pseudo-time motion from the real volume so
that registration can be evaluated against known synthetic translations and
rotations while preserving realistic multiphoton image structure.
