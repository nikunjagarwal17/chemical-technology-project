# Part 2 Quickstart — Code, builds, and benchmarks

This guide lists exact commands and artifacts to run or show during the Part 2 presentation. It assumes you're on Windows and running from the project root.

Prerequisites
- Python 3.10–3.12 recommended for stable local build; CI uses controlled toolchains for wheels.
- A virtual environment (strongly recommended).
- Development dependencies:
  - build, wheel, pip
  - setuptools, numpy, cython (pinned per `pyproject.toml`), pytest, matplotlib

Create and activate a venv

```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

Install dev deps (fast)

```cmd
pip install -r requirements.txt
pip install build cython==0.29.37  # match build pin if required
```

Run tests (pure-Python path)

```cmd
python -m pytest -q
```

Run benchmark script (if present)

```cmd
python -m examples.run_benchmark
```

Attempt a local editable build (optional, may fail on Windows CPython 3.13)

```cmd
pip install -e .
python -c "import simplecantera; print(simplecantera.Reaction)"
```

Create sdist and wheel locally (useful to reproduce CI behaviour)

```cmd
python -m build --sdist --wheel
```

If you have a CI-provided wheel, install and demo

```cmd
pip install dist\simplecantera-<version>-<platform>.whl
python -m test1  # run same demo to show speedup
```

Files to show during Part 2 demo
- `tests/` — run a subset of tests demonstrating numeric correctness
- `examples/` — show `run_benchmark` or `test1/test2/test3` outputs
- CI artifacts (link to workflow run) and `build/` logs

Troubleshooting & notes
- If the local build fails with C-API or linker errors, capture the build logs and prefer CI-built wheels for demonstrations.
- For consistent Cython-generated code, ensure the `cython` version matches `pyproject.toml` pin.

This quickstart is intended for Part 2 live demos showing tests, local attempts to build, and benchmark results.
