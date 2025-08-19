from setuptools import setup, find_packages, Extension
import os

install_requires = ['numpy']

# Try to build Cython extension if Cython is available
extensions = []
try:
    from Cython.Build import cythonize
    # include numpy headers when available (required for Cython wrappers that use numpy)
    include_dirs = [os.path.abspath('src')]
    try:
        import numpy
        include_dirs.append(numpy.get_include())
    except Exception:
        # numpy headers not available at setup-time; build may still work if numpy is
        # installed in the environment used by pip. If build fails with missing
        # 'numpy/arrayobject.h', install numpy in the build environment and retry.
        pass

    # Try to discover Python's libs dir and release import lib name so the linker
    # can find the correct pythonXXX.lib even if a debug import lib (python313t.lib)
    # is not present. This avoids requiring admin to copy files in the Python install.
    try:
        import sysconfig, sys
        libdir = sysconfig.get_config_var('LIBDIR') or os.path.join(sys.exec_prefix, 'libs')
        py_lib_name = f'python{sys.version_info.major}{sys.version_info.minor}'
    except Exception:
        libdir = os.path.join(sys.exec_prefix, 'libs')
        py_lib_name = None

    # prefer project root for library lookup so we can provide a local shim if
    # the Python install doesn't expose a debug import lib (pythonXXXt.lib)
    library_dirs = [os.path.abspath('.')] + ([libdir] if libdir else [])
    ext = Extension(
        'simplecantera._pybindings',
        sources=['simplecantera/pybindings.pyx', 'src/core.cpp'],
        language='c++',
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=[py_lib_name] if py_lib_name else [],
    )
    extensions = cythonize([ext])
except Exception:
    # Cython not available; skip building extension
    extensions = []

setup(
    name='simplecantera',
    version='0.1.0',
    packages=find_packages(),
    install_requires=install_requires,
    description='Minimal Cantera-inspired MVP: reversible A<=>B reactor',
    ext_modules=extensions,
)
