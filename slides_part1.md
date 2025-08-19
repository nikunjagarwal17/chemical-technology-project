% SimpleCantera MVP — Part 1 Slides

# Slide 1 — Title
SimpleCantera MVP — Part 1
Design, architecture, and demo plan

---
# Slide 2 — Project summary
- Goal: lightweight, testable MVP inspired by Cantera
- Layers: pure-Python reference + optional C++ performance core
- Deliverables for Part 1: folder map, API surface, demo plan

---
# Slide 3 — Folder structure (high level)
- `simplecantera/` — public API, pure-Python implementation, Cython sources
- `src/` — C++ core
- `examples/` — YAML specs and demo scripts
- `tests/` — pytest unit tests
- Packaging & CI: `pyproject.toml`, `setup.py`, GitHub Actions

---
# Slide 4 — Architecture diagram
- Public API -> pure-Python ref / compiled bindings -> C++ core
- YAML specs -> builder -> Reactor object -> integrators -> outputs (CSV/plot)

---
# Slide 5 — Demo plan
- Use pure-Python fallback for live demo (no compiler needed)
- Commands:
  - `python -m test1`
  - `python -m test2`
  - `python -m test3`
- Show generated PNGs in `examples/`

---
# Slide 6 — Risks & timeline
- Risk: native build fragility across tool versions
- Mitigation: CI builds via `cibuildwheel`, pin Cython in `pyproject.toml`
- Timeline:
  - Part 2 (1 month): stabilise builds, microbenchmarks, tests
  - Part 3 (release): wheels, docs, benchmarks

---
# Thank you / Next steps
- Questions?
- Next milestones: Part 2 planning and benchmark data collection
