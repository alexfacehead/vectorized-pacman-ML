"""Shared fixtures for parity tests."""

import os
import sys
import pytest

# Add original PacmanML to path for reference engine imports
PACMANML_DIR = os.path.expanduser("~/dev/PacmanML")
if PACMANML_DIR not in sys.path:
    sys.path.insert(0, PACMANML_DIR)

MAZE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "levels", "level_1.txt")
ORIGINAL_MAZE_FILE = os.path.join(PACMANML_DIR, "pacman", "levels", "level_1.txt")
