"""Setup script for fiddler_utils package.

DEPRECATED: This setup.py is kept for backward compatibility only.
All packaging configuration has been moved to pyproject.toml.

To build the package, use modern build tools:
    pip install build
    python -m build

Or with uv:
    uv build
"""

from setuptools import setup

# All configuration is now in pyproject.toml
# This file is kept as a minimal shim for backward compatibility
# with tools that don't yet support PEP 517/518
setup()
