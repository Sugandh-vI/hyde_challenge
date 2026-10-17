"""version_6/v6 with switch sets given by the SWV6_LIST environment variable.

SWV6_LIST is a JSON array of switch dicts; each becomes a class A0, A1, ... (merged over v6's
SWITCHES), so one screen run can compare several variants against one baseline run:

    SWV6_LIST='[{"margin": 0.02}, {"vf": {"hand_synergy": 0.5}}]' \
        python screen.py 400 500000 2 baseagent:Agent swagent:A0 swagent:A1 --clock --cpu_scale 2

An empty SWV6_LIST gives A0 = the default (Step 2) configuration.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load import load_agent  # noqa: E402

_m = load_agent(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v6", "agent.py"),
                "v6_agent_sw")
_CFGS = json.loads(os.environ.get("SWV6_LIST") or "[{}]")

for _i, _cfg in enumerate(_CFGS):
    globals()[f"A{_i}"] = type(f"A{_i}", (_m.Agent,), {"__init__": (lambda c: lambda self: _m.Agent.__init__(self, c))(_cfg)})
del _i, _cfg
Agent = A0
