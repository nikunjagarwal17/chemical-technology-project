from setuptools import setup, find_packages, Extension
import os

install_requires = ['numpy']

# Try to build Cython extension if Cython is available
extensions = []
try:
    from Cython.Build import cythonize
    ext = Extension(
        'simplecantera._pybindings',
        sources=['simplecantera/pybindings.pyx', 'src/core.cpp'],
        language='c++',
        include_dirs=[os.path.abspath('src')],
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
