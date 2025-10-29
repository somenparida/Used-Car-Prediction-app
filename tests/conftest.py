"""pytest configuration helpers.

Ensure the repository root is on sys.path so tests can import local
project modules even when pytest is invoked from a parent directory.
"""
from pathlib import Path
import sys


# tests/ is at <repo_root>/tests; repository root is two levels up from this file
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
# Prepend repo root so `import data_processing` and similar imports work
sys.path.insert(0, str(REPO_ROOT))
