from contextlib import contextmanager
import os
import random
import dask
from dask.distributed import Client
import getpass
import tempfile
import shutil
import pickle

from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import (
    TimingMessager,
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
from cellmap_analyze.util.io_util import TimingMessager
from tqdm import tqdm

import random
import dask.bag as db

from functools import wraps
import os
import pickle
from multiprocessing.dummy import Pool  # thread-based Pool


def with_tqdm(fn):
    @wraps(fn)
    def wrapper(lst, *args, **kwargs):
        N = len(lst)
        # wrap the list in tqdm, but still pass it to fn
        return fn(
            tqdm(
                lst,
                total=N,
                miniters=max(1, N // 10),
                maxinterval=120,
                desc=fn.__name__,
            ),
            *args,
            **kwargs,
        )

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
    if np.any(padding):
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

    with TimingMessager(f"Generating {num_blocks} blocks", logger):

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
    approximate_npartitions = min(elements, num_workers * scaling)
    elements_per_worker = elements // approximate_npartitions
    actual_partitions = elements // elements_per_worker
    return actual_partitions


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
        )
    else:
        if cluster_type == "lsf":
            from dask_jobqueue import LSFCluster

            cluster = LSFCluster(
                job_script_prologue=job_script_prologue,
                scheduler_options={
                    "service_kwargs": {
                        "dashboard": {
                            "session_token_expiration": 60
                            * 60
                            * 24,  # 24 hours timeout
                        }
                    },
                },
            )
        elif cluster_type == "slurm":
            from dask_jobqueue import SLURMCluster

            cluster = SLURMCluster()
        elif cluster_type == "sge":
            from dask_jobqueue import SGECluster

            cluster = SGECluster()
        cluster.scale(num_workers)
    try:
        with TimingMessager(
            f"Starting {cluster_type} dask cluster for {msg} with {num_workers} workers",
            logger,
        ):
            client = Client(cluster)
        dashboard_link = client.cluster.dashboard_link
        if cluster_type == "local":
            dashboard_link = dashboard_link.replace("127.0.0.1", hostname)
        print_with_datetime(f"Check {dashboard_link} for: {msg} status.", logger)
        yield client
    finally:
        client.shutdown()
        client.close()


def _load_dask_config():
    """Read ``dask-config.yaml`` from the cwd into a plain dict."""
    with open("dask-config.yaml") as f:
        return yaml.load(f, Loader=SafeLoader)


def run_with_oom_retry(
    work_fn,
    num_workers,
    phase_name,
    logger,
    max_retries=3,
    retry_on_oom=True,
    config=None,
):
    """Run a dask phase, halving processes-per-slot and retrying on worker OOM.

    Ported from mesh-n-bone. ``work_fn`` is called as ``work_fn(workers, config)``
    and must start its own cluster via ``start_dask(workers, ..., config=config)``.
    On ``distributed.scheduler.KilledWorker`` (the typical symptom of a SIGKILL
    from the OS OOM-killer or LSF mem limit), we halve ``processes`` per slot in
    the in-memory dask config and halve ``num_workers`` to match — doubling the
    memory per worker while keeping total slot/CPU budget constant — then retry.

    ``config`` is an optional pre-loaded dask config (e.g. from
    :func:`plan_memory_waves`); when omitted, we read ``dask-config.yaml``
    from cwd. Pass the wave's tuned config when running per-wave so retries
    halve from the wave's processes count rather than the disk default.

    No-ops (passes through) for synchronous (num_workers <= 1) runs, when
    ``retry_on_oom=False``, when ``max_retries < 1``, or for cluster types that
    don't expose a processes-per-slot lever (e.g. local).
    """
    if not retry_on_oom or max_retries < 1 or num_workers <= 1:
        return work_fn(num_workers, config)

    try:
        from distributed.scheduler import KilledWorker
    except ImportError:
        return work_fn(num_workers, config)

    if config is None:
        config = _load_dask_config()

    cluster_type = next(iter(config.get("jobqueue", {})), None)
    if cluster_type not in ("lsf", "slurm", "sge"):
        return work_fn(num_workers, config)

    workers = num_workers
    for attempt in range(max_retries + 1):
        try:
            return work_fn(workers, config)
        except KilledWorker as e:
            processes = int(config["jobqueue"][cluster_type].get("processes", 1) or 1)
            if attempt >= max_retries:
                logger.error(
                    "Phase '%s' hit worker OOM and exhausted %d retries "
                    "(processes/slot=%d). Increase per-slot memory in "
                    "dask-config.yaml.",
                    phase_name, max_retries, processes,
                )
                raise
            if processes <= 1:
                logger.error(
                    "Phase '%s' hit worker OOM with processes/slot already at "
                    "1 — cannot halve further. Increase memory per slot directly.",
                    phase_name,
                )
                raise
            new_processes = max(1, processes // 2)
            new_workers = max(1, workers // 2)
            last_worker = getattr(e, "last_worker", None)
            logger.warning(
                "Phase '%s' worker OOM (retry %d/%d). Halving processes/slot "
                "%d→%d and workers %d→%d to double memory per worker. "
                "Last worker: %s",
                phase_name, attempt + 1, max_retries,
                processes, new_processes, workers, new_workers,
                last_worker or "(unknown)",
            )
            config["jobqueue"][cluster_type]["processes"] = new_processes
            workers = new_workers


# ---------------------------------------------------------------------------
# Memory-aware wave scheduling (ported from mesh-n-bone).
#
# Group variable-cost work items into "waves" whose per-worker memory class
# differs. Each wave runs as its own dask cluster with processes/slot tuned
# to fit one item per worker. Total slot/CPU budget stays constant across
# waves; what changes is how the per-slot memory is sliced.
# ---------------------------------------------------------------------------


@dataclass
class WavePlan:
    processes: int
    workers: int
    item_ids: list  # opaque to dask_util; whatever the caller passed in
    max_estimated_peak_bytes: int
    config: dict | None


def _jobqueue_settings(config):
    """Return ``(cluster_type, settings)`` for a dask-jobqueue config."""
    if not config:
        return None, None
    jobqueue = config.get("jobqueue", {}) or {}
    if not jobqueue:
        return None, None
    cluster_type, settings = next(iter(jobqueue.items()))
    return cluster_type, settings


def _job_memory_bytes(settings):
    """Parse job memory bytes from a dask-jobqueue settings dict."""
    if not settings or "memory" not in settings:
        return None
    from dask.utils import parse_bytes
    return int(parse_bytes(str(settings["memory"])))


def set_jobqueue_processes(config, cluster_type, processes):
    """Set worker processes in a jobqueue config, keeping one thread/process."""
    processes = max(1, int(processes))
    settings = config["jobqueue"][cluster_type]
    settings["processes"] = processes
    if "cores" in settings:
        # dask-jobqueue derives threads per worker from cores/processes.
        # Hold the ratio at 1 so a memory-heavy worker doesn't run multiple
        # items concurrently in-process.
        settings["cores"] = processes


def _recommended_processes(
    estimated_peak_bytes,
    job_memory_bytes,
    base_processes,
    memory_fraction=0.60,
):
    """Pick processes/job so one item fits per process worker."""
    base_processes = max(1, int(base_processes))
    if not job_memory_bytes or estimated_peak_bytes <= 0:
        return base_processes
    usable_job_bytes = int(job_memory_bytes * memory_fraction)
    processes = usable_job_bytes // int(estimated_peak_bytes)
    return max(1, min(base_processes, int(processes)))


def balanced_batches(items, max_batches):
    """Greedy heaviest-first bin-packing of ``items`` into at most
    ``max_batches`` batches, balanced by ``estimated_peak_bytes``.

    ``items`` is an iterable of ``(item_id, estimated_peak_bytes)`` tuples.
    Returns a list of lists of ``item_id`` (heaviest batch first).
    """
    import heapq

    items = list(items)
    if not items:
        return []
    max_batches = max(1, min(int(max_batches), len(items)))
    heap = [(0, i, []) for i in range(max_batches)]
    for item_id, weight in sorted(items, key=lambda p: p[1], reverse=True):
        bin_weight, index, bin_items = heapq.heappop(heap)
        bin_items = bin_items + [item_id]
        heapq.heappush(heap, (bin_weight + max(int(weight), 1), index, bin_items))

    batches = [(weight, bin_items) for weight, _, bin_items in heap if bin_items]
    batches.sort(key=lambda p: p[0], reverse=True)
    return [bin_items for _, bin_items in batches]


def plan_memory_waves(
    items,
    requested_workers,
    config=None,
    batches_per_worker=4,
    memory_fraction=0.60,
):
    """Group ``items`` into waves with memory-aware dask configs.

    Parameters
    ----------
    items : iterable of ``(item_id, estimated_peak_bytes)``
    requested_workers : int
        Initial worker count from the caller (``num_workers``).
    config : dict, optional
        Loaded ``dask-config.yaml``. When omitted (or no jobqueue section),
        a single all-items wave with ``processes=base_processes`` is returned.
    batches_per_worker : int
        Target ratio of bag partitions to workers within a wave.
    memory_fraction : float
        Fraction of per-slot memory considered usable (the rest is dask
        overhead, OS, libraries).

    Returns
    -------
    list of WavePlan, sorted by ``processes`` ascending (high-memory waves first).
    """
    items = list(items)
    if not items:
        return []

    cluster_type, settings = _jobqueue_settings(config)
    base_processes = int((settings or {}).get("processes", 1) or 1)
    job_memory = _job_memory_bytes(settings)

    by_processes = {}
    for item_id, peak_bytes in items:
        procs = _recommended_processes(
            peak_bytes, job_memory, base_processes,
            memory_fraction=memory_fraction,
        )
        by_processes.setdefault(procs, []).append((item_id, peak_bytes))

    waves = []
    requested_workers = max(1, int(requested_workers))
    for procs in sorted(by_processes):
        wave_items = by_processes[procs]
        # Scale worker count inversely to processes/slot: fewer processes
        # per slot ⇒ each item gets more memory ⇒ we want more slots to
        # keep total parallelism roughly stable. Cap by item count + procs.
        nominal_workers = max(
            1, requested_workers * int(procs) // max(1, base_processes)
        )
        max_batches = max(1, nominal_workers * int(batches_per_worker))
        batches = balanced_batches(wave_items, max_batches)
        # Flatten batches back to a single ordered list — cellmap-analyze's
        # compute_blockwise_partitions doesn't take pre-batched input;
        # ordering items by their batch keeps heaviest-first within the wave.
        item_ids = [item_id for batch in batches for item_id in batch]
        workers = min(nominal_workers, max(len(batches), int(procs)))
        if config is not None and cluster_type is not None:
            wave_config = _deepcopy_config(config)
            set_jobqueue_processes(wave_config, cluster_type, procs)
        else:
            wave_config = None
        waves.append(
            WavePlan(
                processes=int(procs),
                workers=int(max(1, workers)),
                item_ids=item_ids,
                max_estimated_peak_bytes=max(p for _, p in wave_items),
                config=wave_config,
            )
        )
    return waves


def _deepcopy_config(config):
    import copy
    return copy.deepcopy(config)


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


def delete_chunks(block_index, get_delete_path_fn, idi_or_location, depth):
    delete_path = get_delete_path_fn(block_index, idi_or_location, depth)
    if os.path.exists(delete_path):
        if os.path.isfile(delete_path):
            os.remove(delete_path)
        else:
            try:
                if os.listdir(delete_path) == []:
                    shutil.rmtree(delete_path, ignore_errors=True)
            except FileNotFoundError:
                # then already removed
                # TODO: Be smarter about deletions
                pass


def delete_tmp_dir_blockwise(
    idi_or_location: ImageDataInterface | str,
    num_workers,
    compute_args,
    is_zarr=True,
    num_blocks=None,
):
    if is_zarr:
        if type(idi_or_location) is str:
            idi = ImageDataInterface(idi_or_location)
        else:
            idi = idi_or_location
        get_delete_path_fn = get_zarr_chunk_path_from_block_index
        num_blocks = get_num_blocks(idi, idi.roi)
        idi_or_location = idi
        basepath, _ = split_dataset_path(idi.path)
        top_level_dir = f"{basepath}/{get_name_from_path(idi.path)}"
    else:
        get_delete_path_fn = get_merge_file_path_from_block_index
        top_level_dir = idi_or_location

    for depth in range(3, 0, -1):
        compute_blockwise_partitions(
            num_blocks,
            num_workers,
            compute_args,
            logger,
            f"deleting temporary dataset at depth {depth}",
            delete_chunks,
            get_delete_path_fn,
            idi_or_location,
            depth,
        )

    shutil.rmtree(top_level_dir)


def get_merge_file_path_from_block_index(block_index, output_dir, depth=3):
    block_string = "/".join(
        [str(block_index // 100**i) for i in range(2, 2 - depth, -1)]
    )
    output_path = f"{output_dir}/{block_string}"
    if depth == 3:
        output_path += ".pkl"

    return output_path


def get_zarr_chunk_path_from_block_index(block_index, idi, depth):
    block = create_block_from_index(idi, block_index)
    # can have duplicates eg 0/0/0 and 0/0/1 go produce the same coords[:2]
    block_coords = block.coords[:depth]
    block_coords_string = "/".join([str(c) for c in block_coords])
    desired_path = f"{idi.path}/{block_coords_string}"
    return desired_path


def write_dask_result_to_pkl(block_index, output_dir, fn, *fn_args, **fn_kwargs):
    """Write a dask block result to a pkl file in the output directory."""
    result = fn(block_index, *fn_args, **fn_kwargs)
    output_path = get_merge_file_path_from_block_index(block_index, output_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(result, f)


def read_results_to_merge(output_dir, num_blocks, threads=None):
    """
    Read 0.pkl…(num_blocks-1).pkl in parallel using a thread pool,
    displaying progress with tqdm.

    Args:
        output_dir: str path where your pickles live
        num_blocks: total number of blocks
        threads:    how many threads to spin up (None = default)

    Returns:
        List of unpickled results in order [0,1,…,num_blocks-1].
    """

    def _load(i):
        path = get_merge_file_path_from_block_index(i, output_dir)
        # 1) Check existence
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing result file: {path}")

        # 2) Check for non-zero size
        size = os.path.getsize(path)
        if size == 0:
            raise RuntimeError(f"Empty/corrupted pickle file (0 bytes): {path}")

        # 3) Try loading
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except EOFError as e:
            # this is the signature of a truncated/invalid pickle
            raise RuntimeError(
                f"EOFError reading {path}: file appears truncated"
            ) from e

    with Pool(threads) as pool:
        # use imap so we can feed it into tqdm
        return list(
            tqdm(
                pool.imap(_load, range(num_blocks)),
                total=num_blocks,
                miniters=max(1, num_blocks // 10),
                maxinterval=120,  # 120 seconds between updates
                desc="Loading blocks",
            )
        )


# STEP 2) Batch elements per partition to amortize overhead
def partition_worker(indices, fn, *worker_args, **worker_kwargs):
    for idx in indices:
        fn(idx, *worker_args, **worker_kwargs)
    return []  # discard results


def partition_worker_with_write(indices, fn, output_dir, *worker_args, **worker_kwargs):
    for idx in indices:
        write_dask_result_to_pkl(idx, output_dir, fn, *worker_args, **worker_kwargs)
    return []  # discard results


def compute_blockwise_partitions(
    num_blocks: int,
    num_workers: int,
    compute_args: dict,
    logger,
    msg: str,
    fn,
    *fn_args,
    randomize: bool = False,
    merge_info=None,
    config: dict | None = None,
    **fn_kwargs,
):
    """
    Partition 0..num_blocks-1 across num_workers, optionally randomize,
    collect all raw partitions onto the driver, shut the cluster down,
    then merge locally with a tqdm progress bar.
    """

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
    if merge_info is None:
        bag = bag.map_partitions(partition_worker, fn, *fn_args, **fn_kwargs)
    else:
        merge_fn, output_directory = merge_info
        os.makedirs(output_directory, exist_ok=True)
        bag = bag.map_partitions(
            partition_worker_with_write, fn, output_directory, *fn_args, **fn_kwargs
        )

    # STEP 3) Spin up Dask, run
    with start_dask(num_workers, msg, logger, config=config):
        with TimingMessager(msg.capitalize(), logger):
            try:
                bag.compute(**compute_args)
            except Exception as e:
                print("Compute raised an exception:", e)
                raise

            if merge_info is None:
                return

    with TimingMessager(f"Reading results for {msg}", logger):
        list_of_results = read_results_to_merge(
            output_directory, num_blocks, threads=len(os.sched_getaffinity(0))
        )

    with TimingMessager(f"Merging results for {msg}", logger):
        merged_results = with_tqdm(merge_fn)(list_of_results)

    delete_tmp_dir_blockwise(
        output_directory,
        num_workers,
        compute_args,
        is_zarr=False,
        num_blocks=num_blocks,
    )

    return merged_results
