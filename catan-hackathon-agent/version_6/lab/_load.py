"""Import one agent file with its helper modules isolated from every other agent in the process.

The kit gives each agent its own process, so `import a2` inside an agent always means "the a2.py next
to me".  In the lab two agents share a process, and both `version_6/base` and `version_6/v6` ship an
`a2.py` (plus `v4net.py`, `feats.py`, `fastcopy.py`): a plain `import a2` would hand the second agent
the first agent's module.  This loader pops those helper names out of `sys.modules`, imports the agent
with its own directory first on `sys.path`, then removes the helpers it just imported and restores
whatever the other agent had.  The agent keeps its own module objects through the references it
already holds, so nothing is reloaded or re-executed later.

    from _load import load_agent
    Agent = load_agent(os.path.join(os.path.dirname(__file__), "..", "v6", "agent.py"), "v6_agent")
"""
import importlib.util
import os
import sys

HELPERS = ("a2", "v4net", "feats", "fastcopy", "winfind", "opening", "fmodel", "openfeat", "vf",
           "f_agent")


def load_agent(agent_path, tag):
    path = os.path.abspath(agent_path)
    here = os.path.dirname(path)
    saved = {}
    for name in HELPERS:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)
    sys.path.insert(0, here)
    try:
        spec = importlib.util.spec_from_file_location(tag, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[tag] = mod
        spec.loader.exec_module(mod)
    finally:
        try:
            sys.path.remove(here)
        except ValueError:
            pass
        for name in HELPERS:                      # this agent's helpers leave the shared namespace
            if name in sys.modules and name not in saved:
                sys.modules.pop(name)
        sys.modules.update(saved)                 # the other agent's helpers come back
    return mod
