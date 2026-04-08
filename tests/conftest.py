import pytest
import numpy as np


@pytest.fixture(
    autouse=True,
    scope="session",
    params=[
        (8, 8, 8),  # isotropic
        (3, 7, 5),  # anisotropic - all axes different, non-multiples
        (3.54, 4, 4.5),  # non-integer voxel sizes
    ],
    ids=["isotropic", "anisotropic", "non_integer"],
)
def voxel_size(request):
    """Voxel size in (Z, Y, X) order."""
    return np.array(request.param)


@pytest.fixture(
    params=[
        ((8, 8, 8), (5, 5, 5)),       # 1.6x upsampling
        ((10, 10, 10), (16, 16, 16)), # 0.625x downsampling
        ((8, 10, 5), (5, 4, 5)),      # Mixed anisotropic (1.6, 2.5, 1.0)
    ],
    ids=["upsampling_1.6x", "downsampling_0.625x", "anisotropic_mixed"],
)
def non_integer_voxel_sizes(request):
    """Voxel size pairs that result in non-integer scale factors."""
    return request.param


@pytest.fixture(scope="session")
def image_shape():
    return np.array((11, 11, 11))


@pytest.fixture(scope="session")
def chunk_size():
    return np.array((4, 4, 4))


@pytest.fixture(scope="session")
def shared_tmpdir(tmpdir_factory):
    """Create a shared temporary directory for all test functions."""
    return tmpdir_factory.mktemp("tmp")
