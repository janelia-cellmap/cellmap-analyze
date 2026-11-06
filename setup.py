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
        "cellmap_analyze.cythonizing.touching",
        ["src/cellmap_analyze/cythonizing/touching.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "cellmap_analyze.cythonizing.centers",
        ["src/cellmap_analyze/cythonizing/centers.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        "cellmap_analyze.cythonizing.topological_thinning",
        ["src/cellmap_analyze/cythonizing/topological_thinning.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11", "-O3"],
    ),
    Extension(
        "cellmap_analyze.cythonizing.geodesic",
        ["src/cellmap_analyze/cythonizing/geodesic.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11", "-O3"],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
        },
        language_level=3,
    ),
)
