# # from setuptools import setup
# # from Cython.Build import cythonize

# # setup(
# #     ext_modules=cythonize(["bresenham3D.pyx", "process_arrays.pyx"]),
# #     # extra_compile_args=["-O3", "-std=c11"],
# # )
# from setuptools import setup, Extension
# from Cython.Build import cythonize

# extensions = [
#     Extension("cythonizing.bresenham3D", ["cythonizing/bresenham3D.pyx"]),
#     Extension("cythonizing.process_arrays", ["cythonizing/process_arrays.pyx"]),
# ]

# setup(
#     name="cythonizing",
#     version="0.1",
#     packages=["cythonizing"],
#     ext_modules=cythonize(extensions),
#     zip_safe=False,
# )
