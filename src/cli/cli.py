import os
from cellmap_analyze.util import dask_util, io_util
import logging
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
        if args.num_workers is not None:
            self.run_config["num_workers"] = args.num_workers


def connected_components():
    from cellmap_analyze.process.connected_components import ConnectedComponents

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        cc = ConnectedComponents(**rp.run_config)
        cc.get_connected_components()


def clean_connected_components():
    from cellmap_analyze.process.clean_connected_components import (
        CleanConnectedComponents,
    )

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        ccc = CleanConnectedComponents(**rp.run_config)
        ccc.clean_connected_components()


def contact_sites():
    from cellmap_analyze.process.contact_sites import ContactSites

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        contact_sites = ContactSites(**rp.run_config)
        contact_sites.get_contact_sites()


def mutex_watershed():
    from cellmap_analyze.process.mutex_watershed import MutexWatershed

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        mws = MutexWatershed(**rp.run_config)
        mws.get_connected_components()


def watershed_segmentation():
    from cellmap_analyze.process.watershed_segmentation import (
        WatershedSegmentation,
    )

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        ws = WatershedSegmentation(**rp.run_config)
        ws.get_watershed_segmentation()


def filter_ids():
    from cellmap_analyze.process.filter_ids import FilterIDs

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        filter_ids = FilterIDs(**rp.run_config)
        filter_ids.get_filtered_ids()


def label_with_mask():
    from cellmap_analyze.process.label_with_mask import LabelWithMask

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        lwm = LabelWithMask(**rp.run_config)
        lwm.get_label_with_mask()


def measure():
    from cellmap_analyze.analyze.measure import Measure

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        measure = Measure(**rp.run_config)
        measure.get_measurements()


def fit_lines_to_segmentations():
    from cellmap_analyze.analyze.fit_lines_to_segmentations import (
        FitLinesToSegmentations,
    )

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        fit_lines = FitLinesToSegmentations(**rp.run_config)
        fit_lines.get_fit_lines_to_segmentations()


def assign_to_cells():
    from cellmap_analyze.analyze.assign_to_cells import AssignToCells

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        atc = AssignToCells(**rp.run_config)
        atc.get_cell_assignments()
