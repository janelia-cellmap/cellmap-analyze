from contextlib import contextmanager
import os
import random
import dask
from dask.distributed import Client
import getpass
import tempfile
import shutil


from cellmap_analyze.util.image_data_interface import ImageDataInterface
from .io_util import (
    Timing_Messager,
    print_with_datetime,
    split_dataset_path,
    get_name_from_path,
)
from datetime import datetime
import yaml
from yaml.loader import SafeLoader
from dataclasses import dataclass
from funlib.geometry import Coordinate, Roi
import numpy as np
import logging
from contextlib import contextmanager, nullcontext
import dask.bag as db
from cellmap_analyze.util.io_util import Timing_Messager
from tqdm import tqdm

import random
import dask.bag as db

from functools import wraps
from tqdm import tqdm


def with_tqdm(fn):
    @wraps(fn)
    def wrapper(lst, *args, **kwargs):
        # wrap the list in tqdm, but still pass it to fn
        return fn(tqdm(lst, desc=fn.__name__), *args, **kwargs)

    return wrapper


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DaskBlock:
    index: int
    id: int
    full_block_size: Coordinate
    coords: tuple
    read_roi: Roi
    write_roi: Roi
    relabeling_dict: dict


def get_global_block_id(
    roi_shape_voxels: Coordinate, block_roi: Roi, voxel_size: Coordinate
):
    block_start_voxels = block_roi.get_begin() / voxel_size
    id = (
        roi_shape_voxels[0] * roi_shape_voxels[1] * block_start_voxels[2]
        + roi_shape_voxels[0] * block_start_voxels[1]
        + block_start_voxels[0]
        + 1
    )
    return id


def create_block(
    roi,
    block_begin,
    block_size,
    voxel_size,
    padding,
    index,
    read_beyond_roi=True,
    padding_direction="both",
):
    roi_shape_voxels = roi.shape / voxel_size
    write_roi = Roi(block_begin, block_size).intersect(roi)
    block_id = get_global_block_id(roi_shape_voxels, write_roi, voxel_size)
    read_roi = write_roi
    if padding:
        amount_neg = 0
        amount_pos = 0
        if padding_direction in ["both", "neg", "pos"]:
            amount_neg = padding * int(padding_direction in ["both", "neg"])
            amount_pos = padding * int(padding_direction in ["both", "pos"])
        # in some cases we want to add padding in the positive direction if we need context from outside the roi, for hole finding for example
        elif padding_direction == "neg_with_edge_pos":
            amount_neg = padding
            test_grow = write_roi.grow(amount_pos=padding)
            if test_grow != test_grow.intersect(roi):
                amount_pos = padding
        elif padding_direction == "pos_with_edge_neg":
            amount_pos = padding
            test_grow = write_roi.grow(amount_neg=padding)
            if test_grow != test_grow.intersect(roi):
                amount_neg = padding
        read_roi = write_roi.grow(amount_neg=amount_neg, amount_pos=amount_pos)

    if not read_beyond_roi:
        read_roi = read_roi.intersect(roi)

    coords = (write_roi.begin - roi.begin) / block_size

    return DaskBlock(
        index,
        block_id,
        block_size,
        coords,
        read_roi,
        write_roi,
        {},
    )


def create_block_from_index(
    idi: ImageDataInterface,
    index,
    padding=0,
    roi: Roi = None,
    block_size=None,
    read_beyond_roi=True,
    padding_direction="both",
) -> DaskBlock:
    if not roi:
        roi = idi.roi

    if not block_size:
        block_size = idi.chunk_shape * idi.voxel_size
    roi_start = roi.get_begin()
    roi_end = roi.get_end()
    nchunks = np.ceil(np.array(roi_end - roi_start) / block_size).astype(int)
    block_begin = np.array(np.unravel_index(index, nchunks)) * block_size + roi_start
    return create_block(
        roi,
        block_begin,
        block_size,
        idi.voxel_size,
        padding,
        index,
        read_beyond_roi,
        padding_direction,
    )


def get_num_blocks(idi, roi=None, block_size=None):
    if not block_size:
        chunk_shape = idi.chunk_shape
        if len(chunk_shape) == 4:
            chunk_shape = Coordinate(chunk_shape[1:])
        block_size = chunk_shape * idi.voxel_size
    if not roi:
        roi = idi.roi
    num_blocks = int(
        np.prod([np.ceil(roi.shape[i] / block_size[i]) for i in range(len(block_size))])
    )
    return num_blocks


def create_blocks(
    idi: ImageDataInterface,
    roi: Roi = None,
    block_size=None,
    padding=0,
    read_beyond_roi=True,
):
    if not roi:
        roi = idi.roi
    if not block_size:
        block_size = idi.chunk_shape * idi.voxel_size

    # roi = roi.snap_to_grid(ds.chunk_shape * ds.voxel_size)
    num_blocks = get_num_blocks(idi)

    with Timing_Messager(f"Generating {num_blocks} blocks", logger):

        # create an empty list with num_expected_blocks elements
        block_rois = [None] * num_blocks
        index = 0
        roi_start = roi.get_begin()
        roi_end = roi.get_end()
        for z in range(roi_start[0], roi_end[0], block_size[0]):
            for y in range(roi_start[1], roi_end[1], block_size[1]):
                for x in range(roi_start[2], roi_end[2], block_size[2]):
                    block_rois[index] = create_block_from_index(
                        idi, index, padding, roi, block_size, read_beyond_roi
                    )
                    index += 1

        # if index < len(block_rois):
        #     block_rois[index:] = []
    return block_rois


def guesstimate_npartitions(elements, num_workers, scaling=4):
    if not isinstance(elements, int):
        elements = len(elements)
    return min(elements, num_workers * scaling)


def set_local_directory(cluster_type):
    """Sets local directory used for dask outputs

    Args:
        cluster_type ('str'): The type of cluster used

    Raises:
        RuntimeError: Error if cannot create directory
    """

    # From https://github.com/janelia-flyem/flyemflows/blob/master/flyemflows/util/dask_util.py
    # This specifies where dask workers will dump cached data

    local_dir = dask.config.get(f"jobqueue.{cluster_type}.local-directory", None)
    if local_dir:
        return

    user = getpass.getuser()
    local_dir = None
    for d in [f"/scratch/{user}", f"/tmp/{user}"]:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            continue
        else:
            local_dir = d
            dask.config.set({f"jobqueue.{cluster_type}.local-directory": local_dir})

            # Set tempdir, too.
            tempfile.tempdir = local_dir

            # Forked processes will use this for tempfile.tempdir
            os.environ["TMPDIR"] = local_dir
            break

    if local_dir is None:
        raise RuntimeError(
            "Could not create a local-directory in any of the standard places."
        )


@contextmanager
def start_dask(num_workers=1, msg="processing", logger=None, config=None):
    """Context manager used for starting/shutting down dask

    Args:
        num_workers (`int`): Number of dask workers
        msg (`str`): Message for timer
        logger: The logger being used
        config: Overload configuration for dask

    Yields:
        client: Dask client
    """
    job_script_prologue = [
        "export NUMEXPR_MAX_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export NUM_MKL_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export OPENMP_NUM_THREADS=1",
        "export OMP_NUM_THREADS=1",
    ]

    if not config:
        if num_workers == 1:
            # then dont need to startup dask
            with nullcontext():
                yield
                return

        # Update dask
        with open("dask-config.yaml") as f:
            config = yaml.load(f, Loader=SafeLoader)

    cluster_type = next(iter(config["jobqueue"]))
    dask.config.update(dask.config.config, config)
    set_local_directory(cluster_type)

    if cluster_type == "local":
        from dask.distributed import LocalCluster
        import socket

        hostname = socket.gethostname()
        cluster = LocalCluster(
            n_workers=num_workers,
            threads_per_worker=1,
            # job_script_prologue=job_script_prologue,
        )
    else:
        if cluster_type == "lsf":
            from dask_jobqueue import LSFCluster

            cluster = LSFCluster(job_script_prologue=job_script_prologue)
        elif cluster_type == "slurm":
            from dask_jobqueue import SLURMCluster

            cluster = SLURMlsCluster()
        elif cluster_type == "sge":
            from dask_jobqueue import SGECluster

            cluster = SGECluster()
        cluster.scale(num_workers)
        # cluster.adapt(minimum=0, maximum=num_workers)
    try:
        with Timing_Messager(
            f"Starting {cluster_type} dask cluster for {msg} with {num_workers} workers",
            logger,
        ):
            client = Client(cluster)
        dashboard_link = client.cluster.dashboard_link
        if cluster_type == "local":
            dashboard_link = dashboard_link.replace("127.0.0.1", hostname)
        print_with_datetime(f"Check {dashboard_link} for {msg} status.", logger)
        yield client
    finally:
        client.shutdown()
        client.close()


def setup_execution_directory(config_path, logger):
    """Sets up the excecution directory which is the config dir appended with
    the date and time.

    Args:
        config_path ('str'): Path to config directory
        logger: Logger being used

    Returns:
        execution_dir ['str']: execution directory
    """

    # Create execution dir (copy of template dir) and make it the CWD
    # from flyemflows: https://github.com/janelia-flyem/flyemflows/blob/master/flyemflows/bin/launchflow.py
    config_path = config_path[:-1] if config_path[-1] == "/" else config_path
    timestamp = f"{datetime.now():%Y%m%d.%H%M%S}"
    execution_dir = f"{config_path}-{timestamp}"
    execution_dir = os.path.abspath(execution_dir)
    shutil.copytree(config_path, execution_dir, symlinks=True)
    os.chmod(f"{execution_dir}/run-config.yaml", 0o444)  # read-only
    print_with_datetime(f"Setup working directory as {execution_dir}.", logger)

    return execution_dir


def delete_chunks(block_index, idi, depth=3):
    block = create_block_from_index(idi, block_index)
    block_coords = block.coords[
        :depth
    ]  # can have duplicates eg 0/0/0 and 0/0/1 go produce the same coords[:2]
    block_coords_string = "/".join([str(c) for c in block_coords])
    delete_name = f"{idi.path}/{block_coords_string}"
    if os.path.exists(delete_name):
        if os.path.isfile(delete_name):
            os.remove(delete_name)
        elif os.listdir(delete_name) == []:
            shutil.rmtree(delete_name, ignore_errors=True)


def delete_tmp_zarr(
    idi_or_location: ImageDataInterface | str, num_workers, compute_args
):
    if type(idi_or_location) is str:
        idi = ImageDataInterface(idi_or_location)
    else:
        idi = idi_or_location

    num_blocks = get_num_blocks(idi, idi.roi)

    for depth in range(3, 0, -1):
        compute_blockwise_partitions(
            num_blocks,
            num_workers,
            compute_args,
            logger,
            f"deleting temporary zarr dataset at depth {depth}",
            delete_chunks,
            idi,
            depth,
        )

    basepath, _ = split_dataset_path(idi.path)
    shutil.rmtree(f"{basepath}/{get_name_from_path(idi.path)}")


def compute_blockwise_partitions(
    num_blocks: int,
    num_workers: int,
    compute_args: dict,
    logger,
    msg: str,
    fn,
    *fn_args,
    randomize: bool = False,
    merge_fn=None,
    **fn_kwargs,
):
    """
    Partition 0..num_blocks-1 across num_workers, optionally randomize,
    collect all raw partitions onto the driver, shut the cluster down,
    then merge locally with a tqdm progress bar.
    """

    def _partitioner(idxs, *args, **kwargs):
        out = []
        for i in idxs:
            out.append(fn(i, *args, **kwargs))
        return out

    # STEP 1) Build & (optionally) shuffle the bag
    npart = guesstimate_npartitions(num_blocks, num_workers)
    bag = db.range(num_blocks, npartitions=npart)
    if randomize:
        bag = (
            bag.map(lambda i: (random.random(), i))
            .repartition(npartitions=npart)
            .map(lambda pair: pair[1])
        )

    # STEP 2) Map your work fn over each partition
    bag = bag.map_partitions(_partitioner, *fn_args, **fn_kwargs)

    # STEP 3) Spin up Dask, collect ALL partitions to driver
    with start_dask(num_workers, msg, logger) as client:
        with Timing_Messager(msg.capitalize(), logger):
            flat = bag.compute(**compute_args)

    # STEP 4) If no merge_fn: flatten and return raw_parts
    if merge_fn is None:
        return flat

    # STEP 5) If merge_fn: merge the results with a tqdm progress bar
    with Timing_Messager(f"Merging {len(flat)} results", logger):
        result = with_tqdm(merge_fn)(flat)
    return result
