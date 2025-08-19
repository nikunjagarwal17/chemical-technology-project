# Project Folder Line Diagram (workflow-focused)

This diagram lists only the folders directly involved in the development, build, test and release workflow and the key files/contents expected in each folder.

project/                      <- repository root
|
|-- simplecantera/            # Primary Python package (public API + fallback)
|   |-- __init__.py           # import fallback logic (compiled vs pure-Python)
|   |-- purepy.py             # pure-Python reference implementation (thermo, reactions, reactors, integrators)
|   |-- io.py                 # YAML/spec helpers and builders
|   |-- pybindings.pyx        # Cython wrapper source (optional, compiled path)
|   `-- (generated) pybindings.cpp  # may be present during build
|
|-- pyroxa/                   # (alternate package namespace / docs reference)
|   `-- ...                   # additional Python sources used in documentation/examples
|
|-- src/                      # C++ numerical core (performance-critical code)
|   |-- core.h                # C++ declarations for integrators and simulation entry points
|   `-- core.cpp              # C++ implementations (RK4, RK45/Cash–Karp adaptive integrator)
|
|-- examples/                 # Reproducible example specs and demo scripts
|   |-- sample_spec.yaml
|   |-- test1_spec.yaml
|   |-- test2_spec.yaml
|   |-- test3_spec.yaml
|   |-- sample_display.py     # example runner that loads YAML and prints/saves results
|   |-- run_example.py
|   `-- (generated) *.png     # plots produced by demo scripts (move to examples/screenshots/ for slides)
|
|-- tests/                    # pytest unit tests validating behavior
|   |-- test_equilibrium.py
|   |-- test_reactor_network.py
|   |-- test_cstr_pfr.py
|   |-- test_multi_reactor.py
|   |-- test_benchmark.py
|   `-- test_adaptive.py
|
|-- docs/                     # Sphinx or long-form documentation
|   |-- conf.py
|   |-- index.rst
|   `-- usage.rst
|
|-- build/                    # Local build artifacts (temporary)
|   `-- ...                   # compiled objects, intermediate files
|
|-- .github/                  # CI workflows and actions (builds, cibuildwheel runs)
|   `-- workflows/            # GitHub Actions workflow YAMLs
|
`-- (root-level files)
    |-- README.md
    |-- FOLDER_STRUCTURE.md
    |-- PROGRESS.md
    |-- PART1_QUICKSTART.md
    |-- PART2_QUICKSTART.md
    |-- PART3_QUICKSTART.md
    |-- setup.py / pyproject.toml / MANIFEST.in  # packaging metadata
    `-- requirements.txt

Notes
- The `simplecantera/` package is the runtime entrypoint; it attempts to load compiled `_pybindings` (if present) and otherwise uses `purepy.py`.
- `src/` implements the C++ core; changes here require updating `pybindings.pyx` and rebuilding the extension.
- `examples/` and `tests/` are the primary artifacts you should open during demos and presentations — they are reproducible and require no native build.
- `.github/workflows` contains the CI logic that produces wheels (useful for Part 2/3 demos).

If you'd like this as a printable handout or an SVG diagram for slides, I can convert the ASCII tree into a simple diagram image next.
