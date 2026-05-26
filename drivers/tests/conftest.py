"""Pytest configuration for the drivers test suite.

Ensures sys.path and CWD are set to drivers/ before any test module is
imported, matching the assumptions made by python_driver_wrapper.py.
"""
import os
import sys
from pathlib import Path

_DRIVERS_DIR = Path(__file__).parent.parent.resolve()
_REPO_ROOT = _DRIVERS_DIR.parent

for _p in [str(_DRIVERS_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Must be set before test modules are imported; python_driver_wrapper.py does
# sys.path.append("..") at import time, which resolves relative to CWD.
os.chdir(str(_DRIVERS_DIR))

# TMPDIR on the cluster points to /scratch/tmp which doesn't exist locally.
# Clear it here so tempfile.mkdtemp falls back to /tmp for all tests.
# Tests that need a specific TMPDIR set it themselves via monkeypatch.setenv.
os.environ.pop("TMPDIR", None)
