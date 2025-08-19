% Pyroxa MVP — Part 2 Slides

# Slide 1 — Title
Pyroxa MVP — Part 2
Code status, tests, and benchmarks

---
# Slide 2 — What changed since Part 1
- Pure-Python implementation validated with unit tests.
- Examples and scripts added for benchmarking.
- C++ core present; Cython wrappers added, CI pipeline configured.

---
# Slide 3 — Tests & correctness
- Show test summary (pytest) and example outputs.
- Discuss matching results between pure-Python and compiled path.

---
# Slide 4 — Benchmarks
- Approach: microbenchmark MultiReactor on representative network.
- Expected: C++ core yields significant speedups for large problems.

---
# Slide 5 — Build status and logs
- Demonstrate CI artifacts and any remaining local build errors.
- Action plan to resolve remaining build issues.

---
# Slide 6 — Roadmap to Part 3
- Finalize wheels (CI), polish docs, publish release and benchmarks.
- Acceptance criteria and timeline.
