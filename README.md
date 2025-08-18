# SimpleCantera MVP

Minimal MVP inspired by Cantera for a reversible reaction A <=> B in a constant-volume, isothermal reactor.

Quick start

- Install runtime deps (pure-Python mode):

```bash
pip install -r requirements.txt
```

- Run example (pure-Python fallback):

```bash
python -m examples.run_example
```

Package layout

- `simplecantera/` - package code (pure-Python fallback)
- `examples/` - example script
- `tests/` - simple unit test

Notes

- A C++ core and Cython bindings are planned as placeholders; this MVP uses a pure-Python fallback so you can run immediately without compiling.
