"""version_6/base/agent.py (the untouched SUB3 reference), imported in isolation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load import load_agent  # noqa: E402

_m = load_agent(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "base", "agent.py"),
                "base_sub3_agent")
Agent = _m.Agent
