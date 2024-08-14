from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("bresenham3D.pyx"),
    extra_compile_args=["-O3", "-std=c11"],
)
