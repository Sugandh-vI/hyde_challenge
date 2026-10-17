"""friend 2's v4 value-net agent (version_6/v6/v4net.py), imported in isolation; used as an opponent."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load import load_agent  # noqa: E402

_m = load_agent(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v6", "v4net.py"),
                "v4_agent")
Agent = _m.Agent
