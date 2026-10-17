"""Preset P4: thin wrapper around version_3/v3.

Everything (opening, rules, search) lives in v3/v3agent.py; this file only selects the preset so
the evaluation harness can load it as an ordinary agent.  It works both in place (run from the
kit root) and from a zip built with version_3/make_submission.sh (files copied beside agent.py).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_HERE, "..", "..", "v3"), _HERE):
    if os.path.exists(os.path.join(_cand, "v3agent.py")):
        sys.path.insert(0, _cand)
        break

from v3agent import Agent as _Core  # noqa: E402


class Agent:
    def __init__(self):
        self._impl = _Core(preset="P4")

    def decide(self, obs, playable_actions):
        return self._impl.decide(obs, playable_actions)
