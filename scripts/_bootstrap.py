"""Runtime path bootstrap for standalone script execution.

Scientific plotting scripts are often launched directly from IDE run-configurations.
This helper makes `src/` importable without requiring manual PYTHONPATH setup.
"""

import os
import sys


def ensure_src_on_path():
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(scripts_dir)
    src_dir = os.path.join(project_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

