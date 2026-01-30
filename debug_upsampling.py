import numpy as np

# Simulate the downsampling issue
original_shape = (11, 11, 11)
voxel_size_original = np.array([32, 16, 8])
voxel_size_downsampled = np.array([64, 32, 16])

# Create original array
original = np.ones(original_shape)
print(f"Original shape: {original.shape}")

# Downsample with [::2, ::2, ::2]
downsampled = original[::2, ::2, ::2]
print(f"Downsampled shape: {downsampled.shape}")

# Calculate rescale factors
rescale_factors = tuple(vs / ovs for vs, ovs in zip(voxel_size_downsampled, voxel_size_original))
print(f"Rescale factors: {rescale_factors}")

# Upsample using repeat (current method)
upsampled = (
    downsampled.repeat(int(rescale_factors[0]), axis=0)
    .repeat(int(rescale_factors[1]), axis=1)
    .repeat(int(rescale_factors[2]), axis=2)
)
print(f"Upsampled shape: {upsampled.shape}")
print(f"Expected shape: {original_shape}")
print(f"Shape mismatch: {upsampled.shape != original_shape}")

# Calculate what the physical ROI would be
# The physical size of original: (11*32, 11*16, 11*8) = (352, 176, 88) nm
# The physical size of downsampled: (6*64, 6*32, 6*16) = (384, 192, 96) nm
print(f"\nPhysical size of original: {tuple(s*v for s, v in zip(original_shape, voxel_size_original))}")
print(f"Physical size of downsampled: {tuple(s*v for s, v in zip(downsampled.shape, voxel_size_downsampled))}")

# When we upsample to output_voxel_size [32, 16, 8], we'd expect shape based on physical size:
# (384/32, 192/16, 96/8) = (12, 12, 12)
expected_upsampled_from_physical = tuple(int(s*v // ov) for s, v, ov in zip(downsampled.shape, voxel_size_downsampled, voxel_size_original))
print(f"Expected upsampled shape from physical size: {expected_upsampled_from_physical}")
