# Part 3 Quickstart — Release & Installation

This guide lists commands and artifacts to demonstrate final release behaviour (installing wheels, running performance demos) during Part 3.

Prerequisites
- A clean machine or virtual environment to validate wheel installation.
- Access to CI artifacts (wheels) or `dist/` containing built wheels.

Install from CI artifact (example)

```cmd
pip install https://ci.example.com/artifacts/simplecantera-1.0.0-cp310-win_amd64.whl
```

Or install from local dist directory

```cmd
pip install dist\simplecantera-<version>-<platform>.whl
```

Validate installation

```cmd
python -c "import simplecantera; print(simplecantera.Reaction)"
python -m test1  # run demo to validate correct behaviour and performance
```

Benchmark & compare

```cmd
# before: pure-Python run
python -m examples.run_benchmark --mode=python
# after: compiled core run
python -m examples.run_benchmark --mode=compiled
```

Release artifacts to prepare
- Wheels for supported platforms and Python versions
- Changelog and release notes
- Documentation site snapshot

This quickstart is intended to be used in Part 3 to validate the final release and reproduce benchmarks.
