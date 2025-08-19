PART 2 — Handout (Condensed)

Objective
- Demonstrate completed code paths, show test results, benchmarks, and the status of compiled bindings/wheels.

What we'll show
- Unit test summary and a short test run: `python -m pytest tests/test_equilibrium.py -q` (example)
- Benchmark summary: results from `examples/run_benchmark.py` (or provided benchmark script)
- Build attempt logs: show `build/` or CI artifacts; explain any failures and the mitigations.

Key talking points
- The pure-Python reference implementation is the canonical behavior - tests pass locally.
- The C++ core is present and gives a route to performance improvement; wheel builds are handled in CI for reproducibility.
- Remaining tasks: stabilise local builds or rely on CI-produced wheels; add more multi-species reaction tests and benchmarking harness.

Demo commands

```cmd
# run specific test(s)
python -m pytest -q tests/test_reactor_network.py

# run a benchmark (if present)
python -m examples.run_benchmark
```

Acceptance criteria for Part 2
- Tests demonstrating numerical correctness pass.
- Benchmark shows measurable improvement using compiled path (if available) or a plan is presented to obtain CI-built wheels.

Next steps (toward Part 3)
- Finalise wheel builds in CI and publish artifacts.
- Add performance dashboards and finalize docs.
