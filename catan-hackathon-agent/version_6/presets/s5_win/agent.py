"""preset `s5_win`

Step 5. s2_clock plus the depth-3 win finder from actual VP >= 8 (+0.3 pt +-0.3, but 3 of 800 games ran out of clock: NOT recommended).

Self-contained: `core.py` is `version_6/v6/agent.py`, and every helper module, the value-net
weights and the opening model it uses sit in this folder.  This file only chooses the switches.

    python -m catan_kit.evaluate --a version_6/presets/s5_win/agent.py --b examples/f_agent.py --games 1000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core

OVERRIDES = {'winfind_vp': 8}


class Agent(core.Agent):
    def __init__(self):
        core.Agent.__init__(self, OVERRIDES)
