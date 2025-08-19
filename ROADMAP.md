Roadmap & planned MVP improvements

Short-term (MVP):
- Produce binary wheels for supported platforms via CI (cibuildwheel) and upload them as release artifacts.
- Finalize Cython bindings and ensure generated code is compatible with target Python versions (pin Cython appropriately per target).
- Improve unit test coverage for multi-reactor and multi-species scenarios.

Medium-term:
- Add a small YAML/CTI parser compatible with Cantera-style minimal mechanism files and document the supported subset.
- Add more integrators and solver options (explicit/implicit, stiffness-aware solvers via existing libraries when feasible).
- Add small performance benchmarks and CI-run microbenchmarks.

Long-term:
- Add a richer thermodynamics model and equilibrium solver.
- Add more reactor network types and robust IO (CSV, JSON). 
- Prepare and publish stable releases to PyPI with wheels for major platforms.

Notes
-----
- The roadmap prioritizes developer experience (repeatable CI builds and test coverage) and getting correct results over aggressive optimization. Wheels will be produced once CI is stable across the matrix.
