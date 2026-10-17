"""version_6/v6/agent.py (the working copy), imported in isolation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load import load_agent  # noqa: E402

_m = load_agent(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v6", "agent.py"),
                "v6_agent")
Agent = _m.Agent
