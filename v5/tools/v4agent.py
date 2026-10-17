"""Load the friend's v4 agent by file path under a unique module name (avoids 'agent' clashes)."""
import importlib.util
import os
import sys

_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sub3")
if _D not in sys.path:
    sys.path.append(_D)          # for v4's own helpers: feats, fastcopy
_spec = importlib.util.spec_from_file_location("v4_agent_mod", os.path.join(_D, "v4net.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
Agent = _m.Agent
