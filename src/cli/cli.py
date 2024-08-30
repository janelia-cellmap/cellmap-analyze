import os
from cellmap_analyze.util import dask_util, io_util
import logging
import sys
import importlib
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RunProperties:
    def __init__(self):
        args = io_util.parser_params()

        # Change execution directory
        self.execution_directory = dask_util.setup_execution_directory(
            args.config_path, logger
        )
        self.logpath = f"{self.execution_directory}/output.log"
        self.run_config = io_util.read_run_config(args.config_path)
        self.run_config["num_workers"] = args.num_workers


def contact_sites():
    from cellmap_analyze.process.contact_sites import ContactSites

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        contact_sites = ContactSites(**rp.run_config)
        contact_sites.get_contact_sites()


def measure():
    from cellmap_analyze.analyze.measure import Measure

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        measure = Measure(**rp.run_config)
        measure.get_measurements()
