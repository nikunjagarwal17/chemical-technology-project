Project status — SimpleCantera MVP

What works now (implemented)
----------------------------
- Pure-Python numerical core and API in `simplecantera/purepy.py`:
  - Thermodynamics helper (`Thermodynamics`) with constant cp enthalpy/entropy.
  - Simple reversible reaction model `Reaction` and multi-reaction `ReactionMulti`.
  - Reactor implementations: `WellMixedReactor` (well-mixed), `CSTR`, `PFR` (discretized segments), and `MultiReactor` for N species.
  - Fixed-step RK4 integrators and adaptive step-doubling RK4 (for scalar and multi-species versions).
  - High-level builder `build_from_dict()` and runner `run_simulation_from_dict()` that accept simple spec dictionaries.
- Python package wiring (`simplecantera/__init__.py`) that attempts to import a compiled extension and falls back to the pure-Python implementation automatically.
- Examples and tests:
  - `examples/run_example.py` and `examples/run_cstr.py` (examples that use the pure-Python fallback).
  - Unit tests under `tests/` (8 tests) — currently passing locally (`pytest` shows 8 passed).
- Packaging & CI readiness:
  - `setup.py` and `pyproject.toml` updated to handle Cython and NumPy headers when building compiled extension.
  - `DEV.md` describing local build caveats and recommended CI wheel builds.

What is not (yet) available
---------------------------
- Pre-built wheels published to PyPI (CI pipeline can produce them later).
- A stable compiled extension distribution for all target OS/Python versions (work in progress — CI path recommended).
- Full-featured YAML/CTI parser with robust mechanism support — a minimal loader exists in `simplecantera/io.py`.

Quick status summary
--------------------
- Local development: fully usable with the pure-Python fallback.
- Performance: compiled extension (C++ core + Cython) exists in source but building wheels locally is platform-sensitive; CI is recommended to produce cross-platform wheels.
