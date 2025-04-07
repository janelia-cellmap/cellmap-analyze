![CI](https://github.com/janelia-cellmap/cellmap-analyze/actions/workflows/python.yml/badge.svg)
[![codecov](https://codecov.io/gh/janelia-cellmap/cellmap-analyze/branch/refactor_for_release/graph/badge.svg)](https://codecov.io/gh/janelia-cellmap/cellmap-analyze)

# Tools For Analyzing Large 3D Datasets

This repository is a set of tools for processing and analyzing terabyte size 3D segmentation datasets using Dask. Processing tools involve reading in dataset(s) and outputting another - processed - dataset. These processing tools include the calculation of:

1. `Connected Components`: includes thresholding and masking
2. `Clean Connected Components`: tools for cleaning up an existing segmentation
3. `Contact Sites`: includes setting a distance for contact sites
4. `Filling holes`: fills holes in segmentations
5. `Filtering ids`: filters segmented ids to remove unwanted ones.

In addition, there are also tools for analysis of the 3D datasets including:

1. `Measurement`: measures a variety of properties of the segmented ids (volume, surface area, etc.) as well as properties of contact sites (volume, surface area, contacting objects, etc.).
2. `Fitting lines to segmentations`: useful for cylindrical-type objects.
3. `Assigning to cells`: Assigns objects to the cells they are in based on the center of mass lcoations of the cells.

TODO: Include a detailed description of installation and running the code

