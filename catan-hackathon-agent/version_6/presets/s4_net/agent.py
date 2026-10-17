"""preset `s4_net`

Step 4. s2_clock with the net allowed to override the pre-roll knight rule (the only one of the three widened decisions that did not lose: -0.1 pt +-1.7).

Self-contained: `core.py` is `version_6/v6/agent.py`, and every helper module, the value-net
weights and the opening model it uses sit in this folder.  This file only chooses the switches.

    python -m catan_kit.evaluate --a version_6/presets/s4_net/agent.py --b examples/f_agent.py --games 1000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core

OVERRIDES = {'net_on_knight': True}


class Agent(core.Agent):
    def __init__(self):
        core.Agent.__init__(self, OVERRIDES)
