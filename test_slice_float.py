import numpy as np
from funlib.geometry import Coordinate

# Test what happens with float slice indices
snapped_offset = Coordinate((0.0, 0.0, 0.0))
snapped_end = Coordinate((3.5, 4.25, 4.0))  # Non-integer values

snapped_slices = tuple(
    slice(snapped_offset[i], snapped_end[i]) for i in range(3)
)

print(f"snapped_slices: {snapped_slices}")
print(f"Type of slice indices: {type(snapped_slices[0].start)}, {type(snapped_slices[0].stop)}")

# Create a test array
arr = np.ones((4, 5, 4))
print(f"Original array shape: {arr.shape}")

# Try slicing with float indices
try:
    result = arr[snapped_slices]
    print(f"Sliced array shape: {result.shape}")
except Exception as e:
    print(f"Error: {e}")

# Convert to int
int_slices = tuple(
    slice(int(snapped_offset[i]), int(snapped_end[i])) for i in range(3)
)
result = arr[int_slices]
print(f"With int conversion, sliced array shape: {result.shape}")
