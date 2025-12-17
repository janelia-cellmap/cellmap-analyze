import os
import sys
import time
import logging
import traceback
from contextlib import ContextDecorator, contextmanager
from subprocess import Popen, PIPE, TimeoutExpired
from datetime import datetime
import argparse
import yaml
from yaml.loader import SafeLoader
from funlib.geometry import Roi
import re

# Much below taken from flyemflows: https://github.com/janelia-flyem/flyemflows/blob/master/flyemflows/util/util.py
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def split_on_last_scale(string):
    """
    Remove scale suffix from path, only if it's at the end and properly formatted.

    Args:
        string: Path that may end with /sN or be sN where N is a number

    Returns:
        Path with scale removed, or original string if no scale found

    Examples:
        "dataset/s0" -> "dataset"
        "dataset" -> "dataset"
        "s0" -> "" (empty string, scale at root level)
        "/s0" -> "" (empty string, scale with leading slash)
        "my_s0_data/s1" -> "my_s0_data" (doesn't confuse s0 in name with scale)
    """
    # Match /sN at the end (with leading slash)
    match = re.search(r"^(.*?)(/s\d+)$", string)
    if match:
        return match.group(1) if match.group(1) else ""

    # Also match sN at the end WITHOUT leading slash (for root dataset case)
    # This handles the case where dataset path is just "s0"
    match = re.search(r"^(s\d+)$", string)
    if match:
        return ""

    # No scale found
    return string


def get_name_from_path(path):
    _, data_name = split_dataset_path(path)
    if data_name.startswith("/"):
        data_name = data_name[1:]
    data_name = split_on_last_scale(data_name)
    return data_name


def split_dataset_path(dataset_path, scale=None) -> tuple[str, str]:
    """Split the dataset path into the filename and dataset

    Args:
        dataset_path ('str'): Path to the dataset
        scale ('int'): Scale to use, if present

    Returns:
        Tuple of filename and dataset
    """

    # split at .zarr or .n5, whichever comes last
    splitter = (
        ".zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else ".n5"
    )

    filename, dataset = dataset_path.split(splitter)

    # include scale if present
    if scale is not None:
        dataset += f"/s{scale}"

    if dataset.startswith("/"):
        dataset = dataset[1:]

    return filename + splitter, dataset


def get_output_path_from_input_path(input_path, suffix="_output"):
    """
    Generate output path from input path.

    Args:
        input_path: Input dataset path (can be string or Path object)
        suffix: Suffix to append to dataset name

    Returns:
        Output path in format: basepath/dataset_name{suffix}

    Examples:
        "/path/data.zarr/dataset/s0" -> "/path/data.zarr/dataset_output"
        "/path/data.zarr/s0" -> "/path/data_output.zarr" (root dataset, creates new zarr file)
        "/path/data.zarr" -> "/path/data_output.zarr" (no scale, creates new zarr file)
        "/path/data.zarr/nested/dataset/s0" -> "/path/data.zarr/nested/dataset_output"
    """
    # Convert to string and strip trailing slashes to ensure consistent handling
    input_path = str(input_path).rstrip("/")

    output_ds_name = get_name_from_path(input_path)
    output_ds_basepath = split_dataset_path(input_path)[0]

    # Handle empty dataset name (root dataset case like data.zarr/s0 or data.zarr)
    if not output_ds_name:
        # For root datasets, append suffix to the zarr/n5 filename itself
        # "/path/data.zarr" + "_blockwise" -> "/path/data_blockwise.zarr"

        # Find the extension (.zarr or .n5)
        if ".zarr" in output_ds_basepath:
            base, ext = output_ds_basepath.rsplit(".zarr", 1)
            return f"{base}{suffix}.zarr{ext}"
        elif ".n5" in output_ds_basepath:
            base, ext = output_ds_basepath.rsplit(".n5", 1)
            return f"{base}{suffix}.n5{ext}"
        else:
            # Shouldn't happen, but fallback
            return f"{output_ds_basepath}{suffix}"

    return f"{output_ds_basepath}/{output_ds_name}{suffix}"


class TimingMessager(ContextDecorator):
    """Context manager to time operations with aligned messages"""

    # width for aligning 'Started X' and 'Completed X'
    PREFIX_WIDTH = 25

    def __init__(self, base_message: str, logger, final_message: str = "Completed"):
        """
        Args:
            base_message: the descriptive part of your log
            logger: the Python logger to call
        """
        self._base_message = base_message
        self._logger = logger
        self._final_message = final_message

    def __enter__(self):
        first_word = self._base_message.split()[0].lower()
        # prefix without timing
        prefix = f"Started {first_word}"
        # align prefix, then show base_message
        msg = f"{prefix:<{self.PREFIX_WIDTH}}: {self._base_message}..."
        print_with_datetime(msg, self._logger)
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj):
        first_word = self._base_message.split()[0].lower()
        elapsed = time.time() - self._start_time

        if exc_type is not None:
            # An exception was raised in the block
            prefix = f"FAILED {first_word}"
            msg = f"{prefix:<{self.PREFIX_WIDTH}}: {self._base_message}! after {elapsed:.2f}s because {exc_value}"
        else:
            # Normal completion
            prefix = f"{self._final_message} {first_word}"
            msg = (
                f"{prefix:<{self.PREFIX_WIDTH}}: {elapsed:.2f}s, {self._base_message}!"
            )

        print_with_datetime(msg, self._logger)
        # Returning False ensures any exception is propagated
        return False


def print_with_datetime(output, logger, log_type="info"):
    """[summary]

    Args:
        output ([type]): [description]
        logger ([type]): [description]
    """
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    if log_type == "info":
        logger.info(f"{now}: {output}")
    elif log_type == "warning":
        logger.warning(f"{now}: {output}")
    elif log_type == "error":
        logger.error(f"{now}: {output}")
    # force every handler to flush
    for h in logger.handlers:
        h.flush()


def read_run_config(config_path):
    """Reads the run config from config_path and stores them

    Args:
        config_path ('str'): Path to config directory

    Returns:
        Dicts of required_settings and optional_decimation_settings
    """

    def get_roi_from_string(roi_string):
        # roi will look like this ["z_start:z_end", "y_start:y_end", "x_start:x_end"]. split it and convert to tuple
        roi_start = [int(d.split(":")[0]) for d in roi_string]
        roi_ends = [int(d.split(":")[1]) for d in roi_string]

        roi_extents = [int(roi_ends[i] - roi_start[i]) for i in range(len(roi_string))]
        roi = Roi(roi_start, roi_extents)
        return roi

    with open(f"{config_path}/run-config.yaml") as f:
        config = yaml.load(f, Loader=SafeLoader)

    for key in config.keys():
        if "roi" in key:
            config[key] = get_roi_from_string(config[key])

    return config


def parser_params():
    """Parse command line parameters including the config path and number of workers."""

    parser = argparse.ArgumentParser(
        description="Code to convert single-scale (or a set of multi-scale) meshes to the neuroglancer multi-resolution mesh format"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to directory containing run-config.yaml and dask-config.yaml",
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=None,
        help="Number of workers to launch (i.e. each worker is launched with a single bsub command)",
    )

    return parser.parse_args()


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    Context manager.

    Redirects a file object or file descriptor to a new file descriptor.

    Example:
    with open('my-stdout.txt', 'w') as f:
        with stdout_redirected(f):
            print('Writing to my-stdout.txt')

    Motivation: In pure-Python, you can redirect all print() statements like this:

        sys.stdout = open('myfile.txt')

        ...but that doesn't redirect any compiled printf() (or std::cout) output
        from C/C++ extension modules.
    This context manager uses a superior approach, based on low-level Unix file
    descriptors, which redirects both Python AND C/C++ output.

    Lifted from the following link (with minor edits):
    https://stackoverflow.com/a/22434262/162094
    (MIT License)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)

    try:
        if fileno(to) == stdout_fd:
            # Nothing to do; early return
            yield stdout
            return
    except ValueError:  # filename
        pass

    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), "wb") as copied:
        flush(stdout)  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, "wb") as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            flush(stdout)
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def flush(stream):
    try:
        # libc.fflush(None)  # Flush all C stdio buffers
        stream.flush()
    except (AttributeError, ValueError, IOError):
        pass  # unsupported


def fileno(file_or_fd):
    fd = getattr(file_or_fd, "fileno", lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def tee_streams(output_path, append=False):
    """
    Context manager.
    All stdout and stderr will be tee'd to a a file on disk.
    (in addition to appearing on the original stdout streams).

    Note: Stdout and stderr will be merged, in both the tee file and the console.
    """
    if append:
        append = "-a"
    else:
        append = ""

    tee = Popen(
        f"tee {append} {output_path}",
        shell=True,
        stdin=PIPE,
        bufsize=1,
        universal_newlines=True,  # line buffering
        preexec_fn=os.setpgrp,
    )  # Spawn the tee process in its own process group,
    # so it won't receive SIGINT.
    # (Otherwise it might close its input stream too early if the user hits Ctrl+C.)
    try:
        try:
            with stdout_redirected(tee.stdin, stdout=sys.stdout):  # pipe stdout to tee
                with stdout_redirected(
                    sys.stdout, stdout=sys.stderr
                ):  # merge stderr into stdout
                    yield
        finally:
            tee.stdin.close()
            try:
                tee.wait(1.0)
            except TimeoutExpired:
                pass
    except:
        # If an exception was raised, append the traceback to the file
        with open(output_path, "a") as f:
            traceback.print_exc(file=f)
        raise
