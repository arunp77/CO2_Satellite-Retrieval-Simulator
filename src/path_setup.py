"""
path_setup.py
=============
Robust sys.path helper for CO2 Retrieval Simulator notebooks.

Import this at the top of any notebook cell BEFORE importing project modules:

    import path_setup   # noqa  — fixes sys.path automatically

It works regardless of where Jupyter is launched from (project root,
notebooks/ folder, or any other directory) by searching upward from
this file's location until it finds the src/ directory.
"""

import os
import sys

def _find_src() -> str:
    """
    Walk up from this file's directory until we find a folder called 'src'
    that contains 'hitran_model.py'.  Returns the absolute path to src/.
    Raises RuntimeError if not found within 5 levels.
    """
    start = os.path.dirname(os.path.abspath(__file__))
    candidate = start
    for _ in range(6):
        src = os.path.join(candidate, "src")
        if os.path.isdir(src) and os.path.isfile(os.path.join(src, "hitran_model.py")):
            return src
        candidate = os.path.dirname(candidate)
    raise RuntimeError(
        f"Could not locate the 'src/' directory starting from {start}.\n"
        "Make sure path_setup.py lives in the project root or notebooks/ folder."
    )

_src_path = _find_src()
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Also expose a convenient FIGURES_DIR for notebooks to write plots into
_project_root = os.path.dirname(_src_path)
FIGURES_DIR   = os.path.join(_project_root, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def fig_path(filename: str) -> str:
    """Return the absolute path for saving a figure. Creates figures/ if needed."""
    return os.path.join(FIGURES_DIR, filename)
