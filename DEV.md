DEV notes — SimpleCantera

Purpose
-------
This file contains concise developer instructions for running, testing and (optionally) building the compiled extension locally. It captures lessons from local builds and CI so you can get productive quickly.

Quick recommendations
---------------------
- You can develop and run everything using the pure-Python fallback immediately. No compiled extension is required for development or examples.
- If you need the compiled extension for performance, prefer building wheels in CI (cibuildwheel). Building local wheels on Windows + Python 3.13 is fragile due to Cython/CPython C-API changes.

Prerequisites (local development)
--------------------------------
- Python 3.10–3.13 (pure-Python works across these). For local extension builds prefer Python 3.11 or 3.12.
- pip, wheel
- On Windows: Visual Studio Build Tools (MSVC) installed if you will compile extensions locally.

Install runtime deps (pure-Python)
----------------------------------
```bash
python -m pip install -r requirements.txt
```

Run tests and examples
----------------------
```bash
python -m pytest -q
python run_test_import.py    # quick import + tiny run
python -m examples.run_example
```

Build notes (compiled extension)
--------------------------------
- The project contains a C++ core under `src/` and Cython bindings in `simplecantera/pybindings.pyx`.
- Building compiled extensions depends on matching Cython <> CPython compatibility:
  - `pyproject.toml` pins build-time Cython to `>=0.29,<3.0` for CI builds.
  - Locally, if you have Cython 3.x installed, it may generate code that needs CPython internals not present in your interpreter (causes unresolved link symbols).
  - Cython 0.29.x can be incompatible with very new CPython versions (e.g., CPython 3.13) because of changed C APIs.

Local extension build (if you still want to try)
------------------------------------------------
1. Prefer using Python 3.11 or 3.12 locally for compiling the extension (easier compatibility with Cython 0.29).
2. Optionally force a specific Cython locally:
   ```bash
   python -m pip install --user --upgrade "Cython<3.0" numpy
   ```
3. Then try an in-place build:
   ```bash
   python setup.py build_ext --inplace
   ```
4. If that succeeds you can create a wheel:
   ```bash
   python -m pip install build
   python -m build --sdist --wheel
   ```

If you hit linker / API errors
-----------------------------
- Common errors seen:
  - unresolved externals referencing `Py_MergeZeroLocalRefcount`, `PyUnstable_Module_SetGIL`, or `_PyLong_AsByteArray` argument mismatch.
  - These are caused by mismatches between (Cython version) <-> (CPython version) <-> (MSVC headers/runtime).
- Remedies:
  - Use CI (recommended): CI builds in controlled containers use a pinned Cython + matching toolchain; they produce reliable wheels for distribution.
  - Locally, switch Python to a version that is known-good with Cython 0.29 (3.11/3.12) or uninstall local Cython 3.x and install 0.29.x.
  - Alternatively, generate `pybindings.cpp` using a controlled environment with Cython 0.29 and commit it (this forces builds to use the committed C++ source without cythonizing). This approach requires care to keep the generated `.cpp` in sync with the `.pyx`.

CI / wheels (future)
--------------------
- The repo contains a GitHub Actions workflow (cibuildwheel) to build wheels across many platforms. That is the recommended route to produce binary wheels for distribution.

Developer tips
--------------
- Use the pure-Python fallback for fast iteration on the algorithm and tests.
- Only attempt local compilation when you need native performance, and try to match the environment used by CI (same Python patch version + same Cython major line).

Contact
-------
If you want, I can:
- Trigger CI to build wheels and triage failures, or
- Generate a compatible `pybindings.cpp` for a chosen target interpreter and commit it so local build does not cythonize.

End of DEV notes
