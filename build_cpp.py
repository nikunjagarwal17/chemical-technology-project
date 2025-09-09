"""
Alternative build script for C++ extensions with fixed library linking
"""

from distutils.core import setup
from distutils.extension import Extension
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
    # Override the library linking configuration
    import distutils.msvccompiler
    
    # Patch the linker to use our library
    original_link = distutils.msvccompiler.MSVCCompiler.link
    
    def patched_link(self, target_desc, objects, output_filename, output_dir=None,
                     libraries=None, library_dirs=None, runtime_library_dirs=None,
                     export_symbols=None, debug=0, extra_preargs=None,
                     extra_postargs=None, build_temp=None, target_lang=None):
        
        # Add our local directory to library search path
        if library_dirs is None:
            library_dirs = []
        library_dirs = [os.path.abspath('.')] + list(library_dirs)
        
        return original_link(self, target_desc, objects, output_filename, output_dir,
                           libraries, library_dirs, runtime_library_dirs,
                           export_symbols, debug, extra_preargs, extra_postargs,
                           build_temp, target_lang)
    
    distutils.msvccompiler.MSVCCompiler.link = patched_link
    
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
