"""A rules-only agent (no value net) from version_6/base, imported in isolation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load import load_agent  # noqa: E402

_m = load_agent(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "base", "a2.py"),
                "base_a2")
Agent = _m.Agent
