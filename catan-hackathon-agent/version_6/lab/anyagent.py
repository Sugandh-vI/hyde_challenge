"""Any preset: the agent file to load comes from the AGENT_PATH environment variable.

    AGENT_PATH=../presets/v6a/agent.py python fastmatch.py anyagent:Agent f_agent:Agent 100 ...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load import load_agent  # noqa: E402

_path = os.environ.get("AGENT_PATH")
if not _path:
    raise SystemExit("set AGENT_PATH to the agent.py of the preset to load")
if not os.path.isabs(_path):                 # relative to the working directory you run from
    _path = os.path.abspath(_path)
_m = load_agent(_path, os.environ.get("AGENT_TAG", "preset_agent"))
Agent = _m.Agent
