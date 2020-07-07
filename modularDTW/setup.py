from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["modular_dtw.pyx","distance_metrics.pyx"])
)
