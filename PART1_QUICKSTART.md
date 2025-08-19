# Part 1 Quickstart — Run the demo (for the meeting)

This short guide lists exact commands and artifacts to run during your Part 1 presentation. It assumes you're on Windows and running from the project root.

Prerequisites
- Python 3.10+ installed and on PATH
- Optional: a virtual environment (recommended)
- Required Python packages (for the pure-Python demo):
  - numpy
  - pyyaml
  - matplotlib

Create and activate a venv (recommended)

```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

Install minimal dependencies

```cmd
pip install -r requirements.txt
```

Run the demos (pure-Python path)

```cmd
python -m test1
python -m test2
python -m test3
```

Expected outputs
- PNG plots will be generated in `examples/`: `test1_plot.png`, `test2_plot.png`, `test3_plot.png`.
- The scripts print final concentrations to the console.

Files to show during the demo
- `folder str` (or `FOLDER_STRUCTURE.md`) — open this file to show the repo layout.
- `pyroxa/__init__.py` — show import fallback logic.
- `pyroxa/purepy.py` — open a small RK4/adaptive integrator snippet.
- Examples: `examples/test1_spec.yaml` and the generated `examples/test1_plot.png`.

Optional: Capture screenshots
- If you need to include screenshots in slides, run the demo before the meeting and take screenshots of the resulting PNGs. Save under `examples/screenshots/` with names like `part1_demo_1.png`.

Troubleshooting tips
- If a module import fails: ensure the virtual environment is activated and `pip install -r requirements.txt` completed successfully.
- If matplotlib is not available, install it with `pip install matplotlib`.

That's it — these commands and files are enough to demonstrate Part 1 without requiring compiled extensions.
