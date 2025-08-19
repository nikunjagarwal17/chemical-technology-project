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

You can also parse simple mechanism files with `simplecantera.io.parse_mechanism(path)` (YAML or minimal CTI placeholder), and run the CSTR example:

```bash
python -m examples.run_cstr
```

```markdown
# SimpleCantera MVP

Minimal MVP inspired by Cantera for a reversible reaction A <=> B in a constant-volume, isothermal reactor.

Quick start (local, no wheels)

- Install runtime deps (pure-Python mode):

```bash
pip install -r requirements.txt
```

- Run the example (pure-Python fallback):

```bash
python run_test_import.py
```

Key developer commands

- Run the unit tests:

```bash
python -m pytest -q
```

- Start an interactive experiment (import the package):

```python
from simplecantera import Reaction, WellMixedReactor
rxn = Reaction(1.0, 0.5)
r = WellMixedReactor(rxn, A0=1.0, B0=0.0)
times, traj = r.run(0.5, 0.1)
```

Notes on the compiled extension

- The project contains a C++ core and Cython bindings for performance. Building wheels or local extensions requires a compatible Cython/Python toolchain. For immediate local development you do not need to build the extension — the pure-Python fallback (`simplecantera/purepy.py`) is functional and tested.

When you're ready to produce binary wheels for distribution, use the CI workflow (cibuildwheel) which will build platform-specific wheels in controlled environments. This was deferred per current plan.

Package layout

- `simplecantera/` - package code (pure-Python fallback)
- `examples/` - example scripts
- `tests/` - unit tests

Documentation & project overview
- `FOLDER_STRUCTURE.md` — formatted folder map and developer guidance
- `PROGRESS.md` — three-part presentation plan (Part 1, Part 2, Part 3)

```
