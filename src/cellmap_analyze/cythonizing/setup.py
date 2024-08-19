from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["bresenham3D.pyx", "process_arrays.pyx"]),
    # extra_compile_args=["-O3", "-std=c11"],
)
