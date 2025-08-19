# Folder Structure and Project Overview

This file documents the repository layout, the purpose of each folder/file, the modular structure, how pieces work together, and the main technologies used.

## Root (project root)
- `README.md`, `USAGE.md`, `DEV.md`, `DOCS_FULL.md`, `STATUS.md`, `ROADMAP.md`
  - Purpose: high-level documentation, developer notes, usage instructions and roadmap.
- `pyproject.toml`, `setup.py`, `MANIFEST.in`, `requirements.txt`
  - Purpose: build/packaging metadata and build-system requirements (PEP 517/518 and setuptools configuration).
- `setup_bdist_output*.txt`, `run.log`, `build/` (build artifacts)
  - Purpose: build logs and temporary build outputs produced during development.
- `INSTALL_WINDOWS_WHEELS.txt`
  - Purpose: notes about Windows wheel installation issues and guidance (developer-maintained).

## Top-level source and packaging folders
- `simplecantera/`
  - Purpose: Python package that provides the public API.
  - Key files:
    - `__init__.py`: imports compiled bindings if available, otherwise falls back to pure-Python implementation.
    - `purepy.py`: pure-Python implementation (thermo, reaction models, reactors, RK4 & adaptive integrators). This ensures the package is usable without a compiled extension.
    - `io.py`: YAML/spec helpers.
    - `pybindings.pyx` (Cython): Cython wrapper source to expose the C++ core when building compiled extension.
    - (generated) `pybindings.cpp`: sometimes present/generated during build; builds may regenerate this from the .pyx.
- `src/`
  - Purpose: C++ numerical core used for performance-critical computations.
  - Key files: `core.cpp` / `core.h` implementing RK4 and Cash–Karp RK45 adaptive integrators and simulation entry points callable from Cython.

## Examples, docs, tests
- `examples/`
  - Purpose: runnable example specs and scripts (YAML specs, small scripts that call the package, plot results, write CSV).
  - Examples include `sample_spec.yaml`, `sample_display.py`, and `test1_spec.yaml`/`test2_spec.yaml`/`test3_spec.yaml` and generated plots.
- `docs/`
  - Purpose: Sphinx/technical docs and long-form documentation.
- `tests/`
  - Purpose: pytest-based unit tests validating the pure-Python implementation and project behavior (8 tests present and passing locally).

## Build, CI and packaging
- CI-related files (not always visible here): GitHub Actions workflows using `cibuildwheel` to produce manylinux/macos/windows wheels.
- Packaging strategy:
  - Pure-Python fallback: keep `purepy.py` fully functional so users can `pip install` from source without C toolchain.
  - Compiled path: C++ core in `src/` + Cython wrapper in `simplecantera/pybindings.pyx`. `pyproject.toml` contains build-system requirements (numpy, Cython pinned) so isolated builds regenerate wrappers with a compatible Cython.
  - `setup.py` contains conditional logic to cythonize when appropriate and add NumPy include dirs.

## Notes and developer guidance
- For local development without a C toolchain, use the pure-Python path — run examples and tests directly.
- For building wheels or compiled extension locally, ensure matching Cython and Python versions (build isolation via `pyproject.toml` is recommended).
- If you want a committed generated wrapper (`pybindings.cpp`) to avoid local cythonize, make sure it was produced with the target Cython/Python combination; otherwise CI-built wheels are safer.

## Status (short)
- Pure-Python fallback implemented and tested locally (tests pass). C++ core and Cython sources exist but local builds on some Windows/Python combos may require CI or environment tweaks.

If you'd like this as a condensed README section or a printable handout, I can prepare that next.
