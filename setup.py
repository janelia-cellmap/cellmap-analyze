from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "cellmap_analyze.cythonizing.bresenham3D",
        ["src/cellmap_analyze/cythonizing/bresenham3D.pyx"],
    ),
    Extension(
        "cellmap_analyze.cythonizing.process_arrays",
        ["src/cellmap_analyze/cythonizing/process_arrays.pyx"],
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)
