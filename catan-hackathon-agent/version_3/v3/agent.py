"""Submission entry point for version_3.

Edit ``PRESET`` (or pass ``Agent(preset="P4")``) to ship a different configuration:
    P1 opening + value extras, no rules, no search
    P2 P1 + rules (robber / discard / endgame)
    P3 P2 + same-turn search          <- default
    P4 P3 with the mixed opening opponent model
    P5 P3 with the generic opening opponent model
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v3agent import Agent  # noqa: E402,F401
