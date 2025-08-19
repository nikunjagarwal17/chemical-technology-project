# Project Progress — Three-part Presentation Plan

This document mirrors the `progress` file but formatted for easier viewing and linking. It splits the project into three presentation parts (Part 1, Part 2, Part 3) and provides detailed notes, artifacts, required utilities, challenges, implementation status, and a technical Q&A prep to support meeting presentations.

---
## Summary
- **Part 1:** Design & folder-structure ideation, API surface, modular architecture, plan and milestones (present now). 
- **Part 2:** Current codebase status — what is implemented, remaining functions, tests, demos (present in ~1 month).
- **Part 3:** Final product — packaged wheels, CI, documentation, polished examples, benchmarking and release (target after Part 2).

Each part below contains goals, artifacts, talking points, challenges, and Q&A prep.

---
## PART 1 — Design, folder structure, and modular plan (present NOW)

### Goals & deliverables for this meeting
- Explain the repository layout and why each folder exists.
- Walk through the modular architecture (public API, pure-Python reference, C++ core, Cython bindings, IO/specs, examples, tests).
- Present the interface contract (inputs/outputs, error modes) for main components.
- Show a demo plan (what you will demo in later parts) and timeline for the next month.

### Artifacts & utilities to have ready for the meeting
- Slide deck (1–10 slides): project summary, architecture diagram, folder map, APIs, timeline, risks.
- Open the file `folder str` or `FOLDER_STRUCTURE.md` to show the annotated folder map in your repo.
- Simple live demo plan (no compiled extension needed): run one example with pure-Python fallback:

```cmd
python -m test1
python -m test2
python -m test3
```

- A short script to show the public API and how import chooses compiled vs pure-Python (open `simplecantera/__init__.py`).
- A one-slide timeline showing Part 2 and Part 3 milestones (deliverable dates, resources needed).

### Design + modular structure to present (talking points)
- Layered design:
  1. Public API: `simplecantera.__init__` provides `Reaction`, `WellMixedReactor`, `CSTR`, `PFR`, `ReactorNetwork`, etc.
  2. Pure-Python reference: `simplecantera/purepy.py` — fully functional fallback implementing RK4 and adaptive integrators (step-doubling RK4).
  3. Performance core: `src/core.cpp` + `src/core.h` — C++ implementation for RK4 and Cash–Karp RK45 adaptive integrator.
  4. Bindings: `simplecantera/pybindings.pyx` (Cython) — exposes C++ functions to Python; `pybindings.cpp` may be generated at build-time.
  5. IO/Specs: YAML specs in `examples/` and helper `io.py` / `build_from_dict` to construct runtime objects.
  6. Packaging/CI: `pyproject.toml`, `setup.py`, and GitHub Actions + cibuildwheel to produce wheels.

### Risks and mitigations to discuss
- Local builds are fragile across CPython/Cython/NumPy permutations (observed unresolved externals and C API mismatches).
  - Mitigation: Use CI (cibuildwheel) for building wheels across OS/versions; pin Cython in `pyproject.toml` for reproducible cythonization during isolated builds.
- Complexity of keeping pre-generated wrapper (`pybindings.cpp`) consistent with build environment.
  - Mitigation: Prefer regenerating with the build-time Cython in PEP-517 isolated builds; only commit generated wrapper if you lock toolchain.

### Anticipated cross-questions and short answers (Part 1)
- Q: "Can I run the project without a C compiler?"
  - A: Yes — use the pure-Python fallback (`purepy.py`) and run the examples/tests as shown.
- Q: "Why both Python and C++?"
  - A: Rapid development and correctness in Python; performance-critical integrators in C++ for speed. Cython bridges them.
- Q: "How do you ensure reproducible builds?"
  - A: `pyproject.toml` lists build-system requires (numpy, pinned Cython), CI uses isolated environments (cibuildwheel) for consistent artifacts.

---
## PART 2 — Current code, demos, remaining functions (present in ~1 month)

### Goals & deliverables
- Demonstrate completed code paths: pure-Python tests, example simulations, plotting, CSV outputs.
- Show progress on C++ core and Cython bindings; present any compiled artifacts (CI-built wheels) or local build logs.
- Explain remaining functions to implement and planned approach/time estimates.

### What is implemented now (summary)
- Pure-Python engine (`purepy.py`): Thermodynamics, Reaction (A <=> B), ReactionMulti, WellMixedReactor, CSTR, PFR, MultiReactor, ReactorNetwork, RK4 and adaptive integrators — tested and working locally.
- Examples + YAML specs: `examples/` contains multiple specs and scripts (`test1.py`, `test2.py`, `test3.py`) that generate plots and CSV.
- Unit tests: `tests/` contains pytest tests (8 tests) — passing locally.
- C++ core and Cython wrapper: source files present (`src/core.cpp`, `src/core.h`, `simplecantera/pybindings.pyx`), but local builds have produced environment-specific issues; CI integration exists.

### Remaining tasks (explicit)
- Fix and stabilise local/native builds on Windows/CPython 3.13 (resolve C-API mismatches) or document a recommended Python toolchain for local builds.
- Implement any remaining advanced reaction kinetics and multi-reaction features (if on roadmap).
- Add more robust tests for C++/Cython path. Add microbenchmarks comparing Python vs C++ paths.
- Polish documentation and examples.

---
## PART 3 — Final product & release (release presentation)

### Goals & deliverables
- Final release artifacts: wheels for common platforms, polished docs, user guide, extended examples and tutorials, and a small benchmark suite.
- Show CI workflow and reproduce a wheel install and demo on a clean environment.

### Deliverables and acceptance criteria
- Wheels published for Windows/macOS/Linux matching target Python versions.
- Documentation pages and usage examples; quickstart working with `pip install mypackage` (or `pip install dist/wheel-file.whl`).
- Tests passing across the CI matrix; microbenchmarks added to repository.

---
## Challenges we faced, how we are solving them, and what's done so far
- Unpredictable local native builds (Cython/CPython ABI mismatch; missing NumPy headers).
  - Status: documented in `DEV.md` and `INSTALL_WINDOWS_WHEELS.txt`; mitigation: pin Cython in `pyproject.toml`, add NumPy as build-require, prefer CI builds (cibuildwheel).
- Keeping wrapper generation reproducible.
  - Status: removed committed pre-generated `.cpp` when it caused mismatches and prefer PEP-517 regeneration; option to commit generated file if tooling is locked.
- Providing both a friendly Python API and a performant core.
  - Status: implemented pure-Python reference; C++ core present; bridging code and packaging strategy created.

---
## How the project works (technical description, for cross-question prep)
- High-level flow:
  - Users write a YAML spec or use API to create a Reaction and a Reactor object.
  - The builder (`build_from_dict`) constructs objects; reactors expose `run(time_span, dt)` producing `times, traj`.
  - Pure-Python path: `purepy.py` uses explicit RK4 or adaptive step-doubling RK4 for integration; these are straightforward explicit integrators.
  - C++ path: `src/core.cpp` implements RK4 and Cash-Karp RK45 (an embedded Runge–Kutta method with adaptive step size control). The C++ functions are wrapped via Cython and exposed as a compiled module for speed.

---
## Presentation utilities and checklist (what to prepare now)
- Slides (PowerPoint/Google Slides/Markdown) covering architecture, timeline, demo plan, and risks.
- Local demo commands for pure-Python path:

```cmd
python -m test1
python -m test2
python -m test3
```

- Files to point to in the repo during presentation:
  - `folder str` (folder map)
  - `simplecantera/__init__.py` (import fallback logic)
  - `simplecantera/purepy.py` (show RK4/adaptive implementation, Reaction API)
  - `examples/test1_spec.yaml`, `examples/test2_spec.yaml`, `examples/test3_spec.yaml` and `examples/*.png`
  - `tests/` (pytest tests) and `run_test_import.py` for a quick import check

---
## Timeline & recommended immediate actions (for you to prepare Part 2 in a month)
- Next week: Stabilise remaining small feature implementations and write 2–3 additional unit tests for edge cases.
- Two weeks: Add a microbenchmark script and collect performance data on your machine and CI.
- Three weeks: Attempt a CI wheel run (or re-run CI) and capture artifacts; resolve any build errors with focused debugging (if any remain).
- One month: prepare Part 2 slides with code diffs, benchmarks, and a live wheel demo if available.

---
## Closing notes
- The repository is intentionally designed to be usable without a native toolchain; use that advantage for Part 1 demos.
- Save CI/build logs and sample wheel artifacts; they are valuable evidence of cross-platform reproducibility to show in the Part 2/3 presentations.

---

If you want, I can:
- Convert this `progress` file to Markdown and create a short slide deck template.
- Generate the Part 1 slide bullets into a compact PDF or Markdown slides.
- Create a quickstart `PART1_QUICKSTART.md` with exact commands to run during your meeting.

Which follow-up should I do next?
