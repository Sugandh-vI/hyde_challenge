"""Load sub3/agent.py (combined bot) under a unique module name."""
import importlib.util
import os
_spec = importlib.util.spec_from_file_location("sub3_agent_mod", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sub3", "agent.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
Agent = _m.Agent
