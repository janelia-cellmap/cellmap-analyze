from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "cellmap_analyze.cythonizing.bresenham3D",
        ["src/cellmap_analyze/cythonizing/bresenham3D.pyx"],
    ),
    Extension(
        "cellmap_analyze.cythonizing.process_arrays",
        ["src/cellmap_analyze/cythonizing/process_arrays.pyx"],
    ),
    Extension(
        "cellmap_analyze.cythonizing.centers",
        ["src/cellmap_analyze/cythonizing/centers.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)
