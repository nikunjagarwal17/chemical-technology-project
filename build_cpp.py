"""
Alternative build script for C++ extensions with fixed library linking
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import sys

# Set up the extension with custom library configuration
include_dirs = [
    os.path.abspath('src'),
    os.path.abspath('pyroxa'),
]

# Add numpy include if available
try:
    import numpy
    include_dirs.append(numpy.get_include())
except ImportError:
    pass

# Create the extension with our local library directory
ext = Extension(
    'pyroxa._pybindings',
    sources=['pyroxa/pybindings.pyx', 'src/core.cpp'],
    include_dirs=include_dirs,
    library_dirs=[os.path.abspath('.')],  # Our local directory with python313t.lib
    language='c++',
    extra_compile_args=['/std:c++17'],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
)

# Build with custom configuration
if __name__ == "__main__":
    # Modern setuptools approach - library dirs are handled in Extension definition above
    # No need for complex compiler patching with modern setuptools
    
    setup(
        name='pyroxa_cpp',
        ext_modules=cythonize([ext], compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "embedsignature": True
        })
    )
