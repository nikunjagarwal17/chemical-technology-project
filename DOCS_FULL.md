Pyroxa — Full usage & developer documentation
=====================================================

This document explains every line of `USAGE.md`, describes the public API and features, shows all common usage combinations with examples, and provides concrete steps to customize functions and rebuild the project.

1) Line-by-line explanation of `USAGE.md`
----------------------------------------
Below each original line from `USAGE.md` is shown (quoted) followed by a plain-English explanation.

- "Install"
  - Section header: indicates how to install runtime dependencies.

- "```bash\npython -m pip install -r requirements.txt\n```"
  - Command block: run this command in a shell to install the Python packages required by the pure-Python implementation and examples. `requirements.txt` lists `numpy`, `matplotlib`, `pytest`, and `PyYAML`.

- "Quick examples"
  - Section header: short, copyable example snippets follow.

- "- Import and run a simple well-mixed reactor::"
  - Introduces a Python snippet showing how to import types from the package and run a reaction simulation.

- The Python snippet:
  - `from pyroxa import Reaction, WellMixedReactor` — imports the two classes used in the simple example.
  - `rxn = Reaction(kf=1.0, kr=0.5)` — creates a reversible reaction with forward rate constant `kf` and reverse `kr`.
  - `r = WellMixedReactor(rxn, A0=1.0, B0=0.0)` — constructs a reactor using the reaction `rxn` and initial concentrations `A0`, `B0` (defaults exist if omitted).
  - `times, traj = r.run(1.0, 0.1)` — runs the reactor from t=0 to t=1.0 with time step 0.1; returns `times` list and `traj` list of concentration states.
  - `print('times:', times)` and `print('trajectory last:', traj[-1])` — print output to the console.

- "- Use the high-level `build_from_dict` runner..."
  - Introduces the dictionary-driven interface that lets you specify reaction, initial conditions, simulation parameters, and system type using plain Python dicts (or YAML loaded into a dict).

- The `build_from_dict` snippet:
  - `build_from_dict` returns a reactor object and the `sim` dict; you then call `reactor.run(time_span, time_step)`.

- "Plotting"
  - States that plotting is optional and requires `matplotlib`; you can build plots from the returned `times` and `traj` lists.

- "When you need more speed"
  - Explains that the project contains a compiled C++ core and Cython bindings for performance, and points to `DEV.md` for build notes.

- "Files of interest"
  - Points to the main source files: `pyroxa/purepy.py`, `pyroxa/pybindings.pyx`, and `src/core.cpp`.

- "Troubleshooting"
  - Advises to consult `DEV.md` if you encounter build or linker errors.

2) Public API reference and behavior
------------------------------------
This section documents the classes, functions and how to use them. All code references are in `pyroxa/purepy.py` (falling back if compiled extension not present).

Thermodynamics
~~~~~~~~~~~~~~
- Class: `Thermodynamics(cp: float = 29.1)`
  - Purpose: Simple ideal-gas-like thermodynamics with constant heat capacity `cp` (units are arbitrary in the MVP).
  - Methods:
    - `enthalpy(T: float) -> float` — returns `cp * T`.
    - `entropy(T: float) -> float` — returns `cp * ln(T)`; returns `nan` for non-positive `T`.
  - Example:
    ```python
    th = Thermodynamics(cp=29.1)
    H = th.enthalpy(300.0)
    S = th.entropy(300.0)
    ```

Reaction
~~~~~~~~
- Class: `Reaction(kf: float, kr: float)`
  - Purpose: Represents a reversible reaction A <=> B with mass-action rate `rate = kf*[A] - kr*[B]`.
  - Methods:
    - `rate(conc)` — where `conc` is `[A, B]` or any list-like with A at index 0 and B at index 1.
  - Example:
    ```python
    rxn = Reaction(1.0, 0.5)
    r = rxn.rate([1.0, 0.0])  # 1*1.0 - 0.5*0.0 = 1.0
    ```

ReactionMulti
~~~~~~~~~~~~~
- Class: `ReactionMulti(kf, kr, reactants: dict, products: dict)`
  - Purpose: General reaction with stoichiometry expressed as maps from species index to stoichiometric coefficient (integers).
  - Rate law: `kf * product(conc[reactant_idx]**nu) - kr * product(conc[product_idx]**nu)`.
  - Example:
    ```python
    # A + 2B -> C where indices are 0->A, 1->B, 2->C
    rx = ReactionMulti(kf=1.0, kr=0.0, reactants={0:1, 1:2}, products={2:1})
    ```

MultiReactor
~~~~~~~~~~~~
- Class: `MultiReactor(thermo, reactions, species, T=300.0, conc0=None)`
  - Purpose: Simulate N species + M reactions using RK4; `reactions` is a list of `ReactionMulti`.
  - Methods:
    - `run(time_span, time_step)` — returns `(times, traj)`, where `traj` is a list of N-length concentration lists.
    - `run_adaptive(time_span, dt_init, atol, rtol)` — step-doubling adaptive integrator.

WellMixedReactor
~~~~~~~~~~~~~~~~
- Class: `WellMixedReactor` (constructor flexible)
  - Short form constructor: `WellMixedReactor(reaction, A0=..., B0=...)` — convenient for single A<=>B.
  - Full form: `WellMixedReactor(thermo, reaction, T=..., volume=..., conc0=(A0,B0))`.
  - Methods:
    - `step(dt)` — advance one RK4 step (internal use).
    - `run(time_span, time_step)` — fixed-step RK4 integration.
    - `run_adaptive(time_span, dt_init, atol, rtol)` — adaptive step-doubling RK4.

CSTR (Continuous Stirred Tank Reactor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Class: `CSTR(thermo, reaction, T=..., volume=..., conc0=(A0,B0), q=0.0, conc_in=(Ain,Bin))`
  - Adds a flow term: `dC/dt = reaction + (q/V)*(C_in - C)`; uses RK4 for time integration.

PFR (Plug Flow Reactor)
~~~~~~~~~~~~~~~~~~~~~~~
- Class: `PFR(thermo, reaction, T=..., total_volume=1.0, nseg=10, conc0=(A0,B0), q=1.0)`
  - Implementation: discretize into `nseg` CSTR segments and simulate reaction + flow transfer between segments.

ReactorNetwork
~~~~~~~~~~~~~~
- Class: `ReactorNetwork(reactors: list, mode: 'series'|'parallel')`
  - `run(time_span, time_step)` steps each reactor and returns a time-history list of reactor states.

Convenience functions
~~~~~~~~~~~~~~~~~~~~~
- `build_from_dict(spec: dict)` — returns `(reactor, sim)` based on a spec dictionary (see YAML examples below).
- `run_simulation_from_dict(spec, csv_out=None, plot=False)` — builds and runs the simulation then optionally writes CSV and/or plots.
- `run_simulation` — compatibility alias.

3) Spec / YAML format (full) and combinations
--------------------------------------------
The high-level `spec` dictionary (and `examples/sample_spec.yaml`) supports:

- Top-level keys:
  - `reaction`: `{'kf': float, 'kr': float}` — reaction parameters.
  - `initial`: `{'temperature': float, 'conc': {'A': val, 'B': val}}` — initial state.
  - `sim`: `{'time_span': float, 'time_step': float}` — simulation times.
  - `system`: `'WellMixed' | 'CSTR' | 'PFR' | 'series'` — system type.
  - For `CSTR` add `cstr` block with `q` and `conc_in`.
  - For `PFR` add `pfr` block with `nseg`, `q`, `total_volume`.
  - Multi-species support: include `species: ["A","B","C",...]` and `reactions: [ {kf,kr,reactants:{name:nu}, products:{name:nu}}, ... ]`.

Examples of combinations (YAML snippets)
---------------------------------------
- WellMixed single A<=>B:
  ```yaml
  reaction:
    kf: 1.0
    kr: 0.5
  initial:
    conc:
      A: 1.0
      B: 0.0
  sim:
    time_span: 2.0
    time_step: 0.1
  system: WellMixed
  ```

- CSTR with inlet concentrations:
  ```yaml
  system: CSTR
  cstr:
    q: 0.5
    conc_in:
      A: 0.0
      B: 0.0
  ... (reaction/initial/sim as above)
  ```

- PFR (10 segments):
  ```yaml
  system: PFR
  pfr:
    nseg: 10
    q: 1.0
    total_volume: 1.0
  ...
  ```

- Multi-species example (2 reactions):
  ```yaml
  species: ["A","B","C"]
  reactions:
    - kf: 1.0
      kr: 0.0
      reactants: {"A": 1}
      products: {"B": 1}
    - kf: 0.5
      kr: 0.0
      reactants: {"B": 1}
      products: {"C": 1}
  initial:
    conc:
      A: 1.0
      B: 0.0
      C: 0.0
  sim:
    time_span: 1.0
    time_step: 0.001
  ```

4) Examples for all common usage combinations (copyable)
--------------------------------------------------------
- Run a WellMixed reactor (pure-Python):
  ```python
  from pyroxa import Reaction, WellMixedReactor
  rxn = Reaction(1.0, 0.5)
  r = WellMixedReactor(rxn, A0=1.0, B0=0.0)
  times, traj = r.run(2.0, 0.1)
  ```

- Build from YAML file and run: `examples/sample_display.py` reads `examples/sample_spec.yaml` and prints final concentrations. Use this pattern to drive batch runs.

- Multi-species, using builder:
  ```python
  from pyroxa.purepy import build_from_dict
  spec = { ... 'species': [...], 'reactions': [...] }
  reactor, sim = build_from_dict(spec)
  times, traj = reactor.run(sim['time_span'], sim['time_step'])
  ```

- Reactor network (series):
  ```python
  spec_net = {'system':'series', 'reactors':[spec1, spec2, ...]}
  net, sim = build_from_dict(spec_net)
  times, history = net.run(sim['time_span'], sim['time_step'])
  ```

5) Custom editing: how to change and consequences
------------------------------------------------
This section explains how to safely edit functions, what files to change, and what happens after changes.

Goal: modify reaction kinetics, add a new integrator, or add a reactor type.

A. Changing reaction kinetics (fast path)
- Files:
  - `pyroxa/purepy.py` — edit `Reaction.rate()` or add a new `Reaction` subclass.

- Steps:
  1. Edit the `Reaction.rate()` method to the desired form (for example include a temperature dependence).
  2. Update or add unit tests under `tests/` to cover the new rate law.
  3. Run `python -m pytest -q` and validate tests pass.

- Effects:
  - All code using `Reaction` will now use the new kinetics. Verify expected behavior via tests and examples.
  - If the compiled extension is later used, ensure the Cython binding layer is consistent (the Cython code expects the same Python API or direct C++ changes may be required).

B. Adding a new reactor type
- Files to modify/add:
  - `pyroxa/purepy.py` — add a class `MyReactor` that implements `step(self, dt)` and `run(self, time_span, time_step)`.
  - Optionally add builder logic in `build_from_dict()` to accept `system: 'MyReactor'`.
  - Add tests in `tests/` validating behavior.

- Steps:
  1. Implement the reactor class following the existing `WellMixedReactor` pattern.
  2. Add a matching `system` branch in `build_from_dict()` so YAML-driven runs can instantiate it.
  3. Add unit tests and examples.
  4. Run `pytest` to verify.

- Effects:
  - Adding new reactors does not affect existing reactors; however shared utilities (e.g., indexing of species, reaction interface) must match to interoperate.

C. Changing numerical integrator or adding adaptive options
- Files:
  - `pyroxa/purepy.py` for Python-only changes, or
  - `src/core.cpp` and `pyroxa/pybindings.pyx` if you want a compiled high-performance integrator.

- Steps for Python-only integrator changes:
  1. Implement new integrator function (e.g., an implicit method) inside `purepy.py` or in a new module.
  2. Add tests for stability and accuracy.
  3. Run `pytest`.

- Steps to add compiled integrator:
  1. Implement the integrator in C++ (`src/core.cpp`) and declare it in `src/core.h`.
  2. Update `pyroxa/pybindings.pyx` to expose the function to Python (or write a thin wrapper class).
  3. Rebuild using CI or locally (see `DEV.md`).

- Effects:
  - Switching integrators may change stability and accuracy; add tests and tune tolerances.

D. Editing Cython / building compiled extension
- Files commonly edited: `pyroxa/pybindings.pyx`, `setup.py`, `pyproject.toml`, `src/core.cpp`.
- Steps:
  1. Edit `.pyx` or `src/core.cpp`.
  2. If modifying Cython signatures, regenerate C++ by running `python setup.py build_ext --inplace` (locally) or rely on CI to cythonize.
  3. If you regenerate `pybindings.cpp` locally with a Cython version, keep in mind Cython version compatibility; better approach is to let CI (pinned Cython <3.0) generate the wrapper.

- Common pitfalls & what will happen:
  - If you use a locally installed Cython 3.x and your target is Python 3.13, generated C code may reference CPython internals that cause link errors (we saw `Py_MergeZeroLocalRefcount` unresolved). If that happens, either use Cython 0.29.x or run builds in the pinned environment.
  - If you change C++ function signatures but not the `.pyx` wrapper, you'll get compile errors; keep signatures in sync.

6) Testing, linting, and validation steps
----------------------------------------
- Run unit tests:
  ```bash
  python -m pytest -q
  ```
- Run the sample display to validate example usage:
  ```bash
  python examples/sample_display.py
  ```
- If you added compiled changes, run build steps (see `DEV.md`) or push to CI.

7) Troubleshooting common errors and their fixes
------------------------------------------------
- Linker unresolved symbols referencing `PyUnstable_*` or `Py_MergeZeroLocalRefcount`:
  - Cause: generated C code from Cython 3.x references internals incompatible with the local Python/headers.
  - Fix: use Cython <3.0 or run CI (isolated build with pinned tools), or use a different Python target for local builds.

- `_PyLong_AsByteArray` signature mismatch or similar compile-time errors:
  - Cause: Cython-generated calls expect different CPython C-API signatures.
  - Fix: ensure Cython & CPython versions are compatible; prefer CI builds for wheels.

8) Quick reference: commands
---------------------------
Install runtime deps:
```bash
python -m pip install -r requirements.txt
```
Run tests:
```bash
python -m pytest -q
```
Run example:
```bash
python examples/sample_display.py
```
Build wheel locally (if you understand compatibility caveats):
```bash
python -m pip install --user "Cython<3.0" numpy
python setup.py build_ext --inplace
python -m pip install build
python -m build --sdist --wheel
```

If you'd like, I can now:
- Generate a more formal API reference (Markdown per public class and method), or
- Create short notebooks demonstrating combinations, or
- Push these docs to the repo and open a PR.

End of document.
