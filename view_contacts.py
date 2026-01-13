# %%
from cellmap_analyze.util.neuroglancer_util import view_in_neuroglancer

view_in_neuroglancer(
    s1="/tmp/pytest-of-ackermand/pytest-current/tmpcurrent/tmp_8_8_8.zarr/segmentation_1/s0",
    s2="/tmp/pytest-of-ackermand/pytest-current/tmpcurrent/tmp_8_8_8.zarr/segmentation_2/s0",
    cc="/tmp/pytest-of-ackermand/pytest-current/tmpcurrent/tmp_8_8_8.zarr/test_contact_sites_distance_3/s0",
    gt="/tmp/pytest-of-ackermand/pytest-current/tmpcurrent/tmp_8_8_8.zarr/contact_sites_distance_3/s0",
)

# %%
