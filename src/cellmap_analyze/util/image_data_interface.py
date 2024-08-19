import os
import sys
import time
import socket
import smtplib
import getpass
import logging
import traceback
from email.mime.text import MIMEText
from contextlib import ContextDecorator, contextmanager
from subprocess import Popen, PIPE, TimeoutExpired, run as subprocess_run
from datetime import datetime
import argparse
from typing import Tuple
import yaml
from yaml.loader import SafeLoader
import tensorstore as ts
import numpy as np
from funlib.geometry import Coordinate
from funlib.geometry import Roi

from cellmap_analyze.util.io_util import split_dataset_path
from funlib.persistence import open_ds

# Much below taken from flyemflows: https://github.com/janelia-flyem/flyemflows/blob/master/flyemflows/util/util.py
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def get_name_from_path(path):
    _, data_name = split_dataset_path(path)
    if data_name.startswith("/"):
        data_name = data_name[1:]
    data_name = data_name.split("/s")[0]
    return data_name


def open_ds_tensorstore(dataset_path: str, mode="r"):
    # open with zarr or n5 depending
    filetype = (
        "zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else "n5"
    )
    spec = {
        "driver": filetype,
        "kvstore": {"driver": "file", "path": dataset_path},
    }
    if mode == "r":
        dataset_future = ts.open(spec, read=True, write=False)
    else:
        dataset_future = ts.open(spec, read=False, write=True)

    return dataset_future.result()


def to_ndarray_tensorstore(
    dataset,
    roi=None,
    voxel_size=None,
    offset=None,
    output_voxel_size=None,
):
    """Read a region of a tensorstore dataset and return it as a numpy array

    Args:
        dataset ('tensorstore.dataset'): Tensorstore dataset
        roi ('funlib.geometry.Roi'): Region of interest to read

    Returns:
        Numpy array of the region
    """
    if roi is None:
        return dataset.read().result()

    if offset is None:
        offset = Coordinate(np.zeros(roi.dims, dtype=int))

    roi -= offset
    roi /= voxel_size

    # Specify the range
    roi_slices = roi.to_slices()

    domain = dataset.domain
    # Compute the valid range
    valid_slices = tuple(
        [
            slice(max(s.start, inclusive_min), min(s.stop, exclusive_max))
            for s, inclusive_min, exclusive_max in zip(
                roi_slices, domain.inclusive_min, domain.exclusive_max
            )
        ]
    )

    # Create an array to hold the requested data, filled with a default value (e.g., zeros)
    output_shape = [s.stop - s.start for s in roi_slices]

    if not dataset.fill_value:
        fill_value = 0
    padded_data = np.ones(output_shape, dtype=dataset.dtype.numpy_dtype) * fill_value
    padded_slices = tuple(
        slice(valid_slice.start - s.start, valid_slice.stop - s.start)
        for s, valid_slice in zip(roi_slices, valid_slices)
    )

    # Read the region of interest from the dataset
    padded_data[padded_slices] = dataset[valid_slices].read().result()

    if output_voxel_size and output_voxel_size != voxel_size:
        rescale_factor = voxel_size[0] / output_voxel_size[0]
        padded_data = (
            padded_data.repeat(rescale_factor, 0)
            .repeat(rescale_factor, 1)
            .repeat(rescale_factor, 2)
        )
    return padded_data


class ImageDataInterface:
    def __init__(self, dataset_path, mode="r"):
        self.path = dataset_path
        filename, dataset = split_dataset_path(dataset_path)
        self.ds = open_ds(filename, dataset, mode=mode)
        self.ts = open_ds_tensorstore(dataset_path)
        self.voxel_size = self.ds.voxel_size
        self.chunk_shape = self.ds.chunk_shape
        self.roi = self.ds.roi
        self.offset = self.ds.roi.offset
        self.output_voxel_size = self.voxel_size

    def to_ndarray_ts(self, roi=None):
        return to_ndarray_tensorstore(
            self.ts, roi, self.voxel_size, self.offset, self.output_voxel_size
        )

    def to_ndarray_ds(self, roi=None):
        return self.ds.to_ndarray(roi)
