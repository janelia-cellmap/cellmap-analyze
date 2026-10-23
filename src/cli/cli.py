import os
from cellmap_analyze.util import dask_util, io_util
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RunProperties:
    def __init__(self, multiscale=None):
        args = io_util.parser_params()

        # Change execution directory
        self.execution_directory = dask_util.setup_execution_directory(
            args.config_path, logger
        )
        self.logpath = f"{self.execution_directory}/output.log"
        self.run_config = io_util.read_run_config(args.config_path)
        if args.num_workers is not None:
            self.run_config["num_workers"] = args.num_workers
        # Commands that write a volume give it a pyramid afterwards; the
        # run-config's `multiscale` key overrides each command's default.
        self.multiscale = multiscale
        if multiscale is not None:
            self.multiscale = self.run_config.pop("multiscale", multiscale)

    def add_multiscale(self, output_path, num_workers, filled=False):
        """Pyramid the command's output, and its hole-filled copy when it made one."""
        if not self.multiscale:
            return
        from cellmap_analyze.process.multiscale import Multiscale

        output_paths = [output_path]
        if filled:
            output_paths.append(
                io_util.get_output_path_from_input_path(output_path, "_filled")
            )
        for path in output_paths:
            # e.g. connected components with deduplicate_ids writes nothing
            # when it finds no duplicates
            if not os.path.exists(path):
                continue
            # The output was just rewritten, so any levels beside it are stale.
            Multiscale(path, overwrite=True, num_workers=num_workers).create_multiscale()


def connected_components():
    from cellmap_analyze.process.connected_components import ConnectedComponents

    rp = RunProperties(multiscale=True)
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        with io_util.TimingMessager(
            "ConnectedComponents", logger, final_message="Complete success"
        ):
            cc = ConnectedComponents(**rp.run_config)
            cc.get_connected_components()
            rp.add_multiscale(cc.output_path, cc.num_workers, filled=cc.fill_holes)


def clean_connected_components():
    from cellmap_analyze.process.clean_connected_components import (
        CleanConnectedComponents,
    )

    rp = RunProperties(multiscale=True)
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        with io_util.TimingMessager(
            "CleanConnectedComponents", logger, final_message="Complete success"
        ):
            ccc = CleanConnectedComponents(**rp.run_config)
            ccc.clean_connected_components()
            rp.add_multiscale(ccc.output_path, ccc.num_workers, filled=ccc.fill_holes)


def contact_sites():
    from cellmap_analyze.process.contact_sites import ContactSites

    rp = RunProperties(multiscale=True)
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        with io_util.TimingMessager(
            "ContactSites", logger, final_message="Complete success"
        ):
            contact_sites = ContactSites(**rp.run_config)
            contact_sites.get_contact_sites()
            rp.add_multiscale(contact_sites.output_path, contact_sites.num_workers)


def mutex_watershed():
    from cellmap_analyze.process.mutex_watershed import MutexWatershed

    rp = RunProperties(multiscale=True)
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        with io_util.TimingMessager(
            "MutexWatershed", logger, final_message="Complete success"
        ):
            mws = MutexWatershed(**rp.run_config)
            mws.get_connected_components()
            rp.add_multiscale(mws.output_path, mws.num_workers, filled=mws.fill_holes)


def skeletonize():
    from cellmap_analyze.process.skeletonize import Skeletonize

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        with io_util.TimingMessager(
            "Skeletonization", logger, final_message="Complete success"
        ):
            skel = Skeletonize(**rp.run_config)
            skel.skeletonize()


def morphological_operations():
    from cellmap_analyze.process.morphological_operations import (
        MorphologicalOperations,
    )

    rp = RunProperties(multiscale=True)
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        with io_util.TimingMessager(
            "MorphologicalOperations", logger, final_message="Complete success"
        ):
            mo = MorphologicalOperations(**rp.run_config)
            mo.perform_morphological_operation()
            rp.add_multiscale(mo.output_vol.level_path, mo.num_workers)


def fill_holes():
    from cellmap_analyze.process.fill_holes import FillHoles

    rp = RunProperties(multiscale=True)
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        with io_util.TimingMessager(
            "FillHoles", logger, final_message="Complete success"
        ):
            fill_holes = FillHoles(**rp.run_config)
            fill_holes.fill_holes()
            rp.add_multiscale(fill_holes.output_path, fill_holes.num_workers)


def filter_ids():
    from cellmap_analyze.process.filter_ids import FilterIDs

    rp = RunProperties(multiscale=True)
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        with io_util.TimingMessager(
            "FilterIDs", logger, final_message="Complete success"
        ):
            filter_ids = FilterIDs(**rp.run_config)
            filter_ids.get_filtered_ids()
            rp.add_multiscale(filter_ids.output_path, filter_ids.num_workers)


def label_with_mask():
    from cellmap_analyze.process.label_with_mask import LabelWithMask

    rp = RunProperties(multiscale=True)
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        with io_util.TimingMessager(
            "LabelWithMask", logger, final_message="Complete success"
        ):
            lwm = LabelWithMask(**rp.run_config)
            lwm.get_label_with_mask()
            rp.add_multiscale(lwm.output_vol.level_path, lwm.num_workers)


def measure():
    from cellmap_analyze.analyze.measure import Measure

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        with io_util.TimingMessager(
            "Measure", logger, final_message="Complete success"
        ):
            measure = Measure(**rp.run_config)
            measure.get_measurements()


def fit_lines_to_segmentations():
    from cellmap_analyze.analyze.fit_lines_to_segmentations import (
        FitLinesToSegmentations,
    )

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        with io_util.TimingMessager(
            "FitLinesToSegmentations", logger, final_message="Complete success"
        ):
            fit_lines = FitLinesToSegmentations(**rp.run_config)
            fit_lines.get_fit_lines_to_segmentations()


def compute_edt():
    from cellmap_analyze.process.compute_edt import ComputeEDT

    rp = RunProperties(multiscale=False)
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        with io_util.TimingMessager(
            "ComputeEDT", logger, final_message="Complete success"
        ):
            cedt = ComputeEDT(**rp.run_config)
            cedt.calculate_edt()
            rp.add_multiscale(cedt.output_path, cedt.num_workers)


def split_narrow_bridges():
    from cellmap_analyze.process.split_narrow_bridges import SplitNarrowBridges

    rp = RunProperties(multiscale=True)
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        with io_util.TimingMessager(
            "SplitNarrowBridges", logger, final_message="Complete success"
        ):
            snb = SplitNarrowBridges(**rp.run_config)
            snb.split_objects()
            rp.add_multiscale(snb.output_path, snb.num_workers)


def multiscale():
    from cellmap_analyze.process.multiscale import Multiscale

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        with io_util.TimingMessager(
            "Multiscale", logger, final_message="Complete success"
        ):
            ms = Multiscale(**rp.run_config)
            ms.create_multiscale()


def assign_to_organelles():
    from cellmap_analyze.analyze.assign_to_organelles import AssignToOrganelles

    rp = RunProperties()
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        with io_util.TimingMessager(
            "AssignToOrganelles", logger, final_message="Complete success"
        ):
            ato = AssignToOrganelles(**rp.run_config)
            ato.get_organelle_assignments()
