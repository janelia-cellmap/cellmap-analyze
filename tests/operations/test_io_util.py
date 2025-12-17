"""
Tests for io_util path handling functions.
Tests the correct behavior for:
- Root datasets (data.zarr/s0)
- Named datasets (data.zarr/dataset/s0)
- Nested datasets (data.zarr/nested/dataset/s0)
"""

import pytest
from cellmap_analyze.util.io_util import (
    split_on_last_scale,
    split_dataset_path,
    get_name_from_path,
    get_output_path_from_input_path,
)


class TestSplitOnLastScale:
    """Test the split_on_last_scale function."""

    def test_with_trailing_scale(self):
        """Test removing scale at end with leading slash."""
        assert split_on_last_scale("dataset/s0") == "dataset"
        assert split_on_last_scale("nested/dataset/s1") == "nested/dataset"

    def test_without_scale(self):
        """Test paths without scale suffix."""
        assert split_on_last_scale("dataset") == "dataset"
        assert split_on_last_scale("nested/dataset") == "nested/dataset"

    def test_root_scale_with_slash(self):
        """Test scale at root level with leading slash."""
        assert split_on_last_scale("/s0") == ""

    def test_root_scale_without_slash(self):
        """Test scale at root level without leading slash."""
        assert split_on_last_scale("s0") == ""
        assert split_on_last_scale("s1") == ""

    def test_scale_in_name_not_confused(self):
        """Test that s0 within dataset name is not removed."""
        result = split_on_last_scale("my_s0_data/s1")
        assert result == "my_s0_data"


class TestSplitDatasetPath:
    """Test the split_dataset_path function."""

    def test_zarr_with_dataset(self):
        """Test splitting zarr path with dataset name."""
        base, dataset = split_dataset_path("/path/data.zarr/dataset/s0")
        assert base == "/path/data.zarr"
        assert dataset == "dataset/s0"

    def test_zarr_root_dataset(self):
        """Test splitting zarr path with root dataset."""
        base, dataset = split_dataset_path("/path/data.zarr/s0")
        assert base == "/path/data.zarr"
        assert dataset == "s0"

    def test_zarr_nested_dataset(self):
        """Test splitting zarr path with nested dataset."""
        base, dataset = split_dataset_path("/path/data.zarr/nested/dataset/s0")
        assert base == "/path/data.zarr"
        assert dataset == "nested/dataset/s0"

    def test_n5_format(self):
        """Test splitting n5 format paths."""
        base, dataset = split_dataset_path("/path/data.n5/dataset/s0")
        assert base == "/path/data.n5"
        assert dataset == "dataset/s0"

    def test_with_scale_parameter(self):
        """Test adding scale parameter."""
        base, dataset = split_dataset_path("/path/data.zarr/dataset", scale=0)
        assert base == "/path/data.zarr"
        assert dataset == "dataset/s0"


class TestGetNameFromPath:
    """Test the get_name_from_path function."""

    def test_normal_dataset(self):
        """Test extracting name from normal dataset path."""
        name = get_name_from_path("/path/data.zarr/dataset/s0")
        assert name == "dataset"

    def test_root_dataset(self):
        """Test extracting name from root dataset (should be empty)."""
        name = get_name_from_path("/path/data.zarr/s0")
        assert name == ""

    def test_nested_dataset(self):
        """Test extracting name from nested dataset path."""
        name = get_name_from_path("/path/data.zarr/nested/dataset/s0")
        assert name == "nested/dataset"

    def test_without_scale(self):
        """Test extracting name when scale is not present."""
        name = get_name_from_path("/path/data.zarr/dataset")
        assert name == "dataset"

    def test_n5_format(self):
        """Test extracting name from n5 format."""
        name = get_name_from_path("/path/data.n5/dataset/s0")
        assert name == "dataset"


class TestGetOutputPathFromInputPath:
    """Test the get_output_path_from_input_path function."""

    def test_normal_dataset_with_suffix(self):
        """Test generating output path for normal dataset."""
        output = get_output_path_from_input_path(
            "/path/data.zarr/dataset/s0", "_output"
        )
        assert output == "/path/data.zarr/dataset_output"

    def test_trailing_slash_handling(self):
        """Test that trailing slashes are handled correctly."""
        # Root dataset with trailing slash
        output = get_output_path_from_input_path("/path/data.zarr/s0/", "_blockwise")
        assert output == "/path/data_blockwise.zarr"

        # Named dataset with trailing slash
        output = get_output_path_from_input_path(
            "/path/data.zarr/dataset/s0/", "_blockwise"
        )
        assert output == "/path/data.zarr/dataset_blockwise"

        # Zarr path with trailing slash (no scale)
        output = get_output_path_from_input_path("/path/data.zarr/", "_output")
        assert output == "/path/data_output.zarr"

    def test_nested_dataset_with_suffix(self):
        """Test generating output path for nested dataset."""
        output = get_output_path_from_input_path(
            "/path/data.zarr/nested/dataset/s0", "_cleaned"
        )
        assert output == "/path/data.zarr/nested/dataset_cleaned"

    def test_root_dataset_creates_new_zarr(self):
        """Test that root datasets create new zarr files."""
        output = get_output_path_from_input_path("/path/data.zarr/s0", "_blockwise")
        assert output == "/path/data_blockwise.zarr"

    def test_root_dataset_without_scale_creates_new_zarr(self):
        """Test that root datasets without scale create new zarr files."""
        output = get_output_path_from_input_path("/path/data.zarr", "_output")
        assert output == "/path/data_output.zarr"

    def test_root_n5_creates_new_n5(self):
        """Test that root n5 datasets create new n5 files."""
        output = get_output_path_from_input_path("/path/data.n5/s0", "_blockwise")
        assert output == "/path/data_blockwise.n5"

    def test_various_suffixes(self):
        """Test different suffix patterns."""
        # Test with blockwise suffix
        output = get_output_path_from_input_path(
            "/path/data.zarr/dataset/s0", "_blockwise"
        )
        assert output == "/path/data.zarr/dataset_blockwise"

        # Test with output suffix
        output = get_output_path_from_input_path(
            "/path/data.zarr/dataset/s0", "_output"
        )
        assert output == "/path/data.zarr/dataset_output"

        # Test with cleaned suffix
        output = get_output_path_from_input_path(
            "/path/data.zarr/dataset/s0", "_cleaned"
        )
        assert output == "/path/data.zarr/dataset_cleaned"

    def test_comprehensive_root_vs_named_behavior(self):
        """
        Comprehensive test showing the key difference:
        - Root datasets create new zarr files
        - Named datasets create new datasets inside existing zarr
        """
        # Root dataset: creates new zarr file
        root_output = get_output_path_from_input_path(
            "/path/data.zarr/s0", "_blockwise"
        )
        assert root_output == "/path/data_blockwise.zarr"
        assert not root_output.startswith("/path/data.zarr/")

        # Named dataset: creates dataset inside existing zarr
        named_output = get_output_path_from_input_path(
            "/path/data.zarr/dataset/s0", "_blockwise"
        )
        assert named_output == "/path/data.zarr/dataset_blockwise"
        assert named_output.startswith("/path/data.zarr/")


class TestEndToEndScenarios:
    """Integration tests for complete path handling scenarios."""

    def test_connected_components_workflow(self):
        """Test path handling for connected components workflow."""
        # Named dataset scenario
        input_path = "/data/experiment.zarr/raw/s0"
        output_path = get_output_path_from_input_path(input_path, "_segmented")
        blockwise_path = get_output_path_from_input_path(output_path, "_blockwise")

        assert output_path == "/data/experiment.zarr/raw_segmented"
        assert blockwise_path == "/data/experiment.zarr/raw_segmented_blockwise"

        # Root dataset scenario
        root_input = "/data/raw.zarr/s0"
        root_output = get_output_path_from_input_path(root_input, "_segmented")
        root_blockwise = get_output_path_from_input_path(root_output, "_blockwise")

        assert root_output == "/data/raw_segmented.zarr"
        assert root_blockwise == "/data/raw_segmented_blockwise.zarr"

    def test_mutex_watershed_workflow(self):
        """Test path handling for mutex watershed workflow."""
        # Named dataset
        output_path = "/data/experiment.zarr/segmentation/s0"
        blockwise = get_output_path_from_input_path(output_path, "_blockwise")
        cleaned = get_output_path_from_input_path(output_path, "_cleaned")
        filled = get_output_path_from_input_path(output_path, "_filled")

        assert blockwise == "/data/experiment.zarr/segmentation_blockwise"
        assert cleaned == "/data/experiment.zarr/segmentation_cleaned"
        assert filled == "/data/experiment.zarr/segmentation_filled"

        # Root dataset
        root_path = "/data/segmentation.zarr/s0"
        root_blockwise = get_output_path_from_input_path(root_path, "_blockwise")
        root_cleaned = get_output_path_from_input_path(root_path, "_cleaned")

        assert root_blockwise == "/data/segmentation_blockwise.zarr"
        assert root_cleaned == "/data/segmentation_cleaned.zarr"

    def test_contact_sites_workflow(self):
        """Test path handling for contact sites workflow."""
        # Named datasets
        org1_path = "/data/exp.zarr/organelle1/s0"
        org2_path = "/data/exp.zarr/organelle2/s0"
        contacts_output = "/data/exp.zarr/contacts/s0"

        contacts_blockwise = get_output_path_from_input_path(
            contacts_output, "_blockwise"
        )
        assert contacts_blockwise == "/data/exp.zarr/contacts_blockwise"

    def test_watershed_segmentation_workflow(self):
        """Test path handling for watershed segmentation workflow."""
        input_path = "/data/binary.zarr/mask/s0"

        distance_transform = get_output_path_from_input_path(
            input_path, "_distance_transform"
        )
        seeds_blockwise = get_output_path_from_input_path(input_path, "_seeds_blockwise")
        seeds = get_output_path_from_input_path(input_path, "_seeds")

        assert distance_transform == "/data/binary.zarr/mask_distance_transform"
        assert seeds_blockwise == "/data/binary.zarr/mask_seeds_blockwise"
        assert seeds == "/data/binary.zarr/mask_seeds"
