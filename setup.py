from setuptools import setup, find_packages, Extension
import os
import sys

# Minimal runtime requirements
install_requires = ['numpy']

# Base include dirs: our C headers live in src/
include_dirs = [os.path.abspath('src')]

# Try to add numpy include directory if available at setup-time. In PEP517
# builds the build backend will install build-requires (Cython, numpy) first,
# so this usually succeeds there as well.
try:
    import numpy
    include_dirs.append(numpy.get_include())
except Exception:
    # If numpy isn't importable at setup-time, the isolated build will provide
    # headers; leave include_dirs as-is.
    pass

# Helper to pick C++ standard flags across platforms
if sys.platform == 'win32':
    extra_compile_args = ['/std:c++17']
else:
    extra_compile_args = ['-std=gnu++14']

# Build extension: prefer to cythonize the .pyx if Cython is available; if not,
# fall back to a pre-generated .cpp file (kept as a fallback for legacy builds).
extensions = []
try:
    from Cython.Build import cythonize
    sources = ['simplecantera/pybindings.pyx', 'src/core.cpp']
    ext = Extension(
        'simplecantera._pybindings',
        sources=sources,
        language='c++',
        include_dirs=include_dirs + [os.path.abspath('simplecantera')],
        extra_compile_args=extra_compile_args,
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )
    extensions = cythonize([ext], compiler_directives={"language_level": "3"})
except Exception:
    # Last-resort fallback: if Cython not present and no pre-generated .cpp, skip ext build
    if os.path.exists(os.path.join('simplecantera', 'pybindings.cpp')):
        sources = ['simplecantera/pybindings.cpp', 'src/core.cpp']
        ext = Extension(
            'simplecantera._pybindings',
            sources=sources,
            language='c++',
            include_dirs=include_dirs + [os.path.abspath('simplecantera')],
            extra_compile_args=extra_compile_args,
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        )
        extensions = [ext]
    else:
        extensions = []

setup(
    name='simplecantera',
    version='0.1.0',
    packages=find_packages(),
    install_requires=install_requires,
    description='Minimal Cantera-inspired MVP: reversible A<=>B reactor',
    ext_modules=extensions,
)
