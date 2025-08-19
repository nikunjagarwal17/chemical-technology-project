PART 1 — Handout (Condensed)

Goal
- Communicate design, architecture, and demo plan for Pyroxa MVP.

What this project is
- Lightweight MVP inspired by Cantera.
- Layers: pure-Python reference + optional C++ performance core exposed via Cython.

Key user-facing API
- Reaction(kf, kr)
- WellMixedReactor(reaction, A0=..., B0=...)
- CSTR, PFR, ReactorNetwork
- build_from_dict(spec) to construct reactors from YAML specs
- reactor.run(time_span, dt) -> (times, trajectory)

Demo (commands)
- (Windows cmd)
  - python -m test1
  - python -m test2
  - python -m test3
- Output: PNGs in `examples/` and final concentrations printed to console.

Why this design
- Pure-Python for correctness, rapid iteration, and reproducible demos without toolchains.
- C++ core planned for performance-critical paths; optional for end users.
- YAML specs for reproducible experiments and CI-driven tests.

Risks to highlight (short)
- Native build fragility across CPython/Cython/NumPy toolchain versions.
- Wrapper regeneration mismatch (committed generated `.cpp` vs build-time Cython).

Immediate next steps (for Part 2)
- Stabilise local/native builds or rely on CI wheels.
- Add 3–5 unit tests for edge cases and microbenchmarks to compare pure-Python vs C++ core.
- Prepare Part 2 demo showing speedup and build artifacts.

Contact & repo pointers (show these during the presentation)
- `FOLDER_STRUCTURE.md` — folder map
- `PROGRESS.md` — three-part plan
- `PART1_QUICKSTART.md` — commands to run during the meeting

One-line takeaway for audience
- A working, testable pure-Python implementation is ready for demos; compiled performance path is staged and will be demonstrated and benchmarked in Part 2.
