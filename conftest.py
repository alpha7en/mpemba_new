"""Pytest bootstrap: make src/ (library) and scripts/ (drivers) importable from tests."""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
