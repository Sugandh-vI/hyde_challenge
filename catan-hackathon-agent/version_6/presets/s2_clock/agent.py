"""preset `s2_clock`

Step 2. version_6 defaults: the value net on every main turn with an adaptive CPU budget instead of SUB3's fixed NET_MIN_CPU = 0.25 s, the counter-opening search capped at 0.24 s, agent2's planner budget as instance settings.

Self-contained: `core.py` is `version_6/v6/agent.py`, and every helper module, the value-net
weights and the opening model it uses sit in this folder.  This file only chooses the switches.

    python -m catan_kit.evaluate --a version_6/presets/s2_clock/agent.py --b examples/f_agent.py --games 1000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core

OVERRIDES = {}


class Agent(core.Agent):
    def __init__(self):
        core.Agent.__init__(self, OVERRIDES)
