"""preset `s3_vf`

Step 3. s2_clock with buildable_nodes 1000 -> 2000, the largest point estimate of the value-function weight screen (+0.6 pt, CI +-1.3: NOT significant, do not prefer it over s2_clock on strength).

Self-contained: `core.py` is `version_6/v6/agent.py`, and every helper module, the value-net
weights and the opening model it uses sit in this folder.  This file only chooses the switches.

    python -m catan_kit.evaluate --a version_6/presets/s3_vf/agent.py --b examples/f_agent.py --games 1000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core

OVERRIDES = {'vf': {'buildable_nodes': 2000.0}}


class Agent(core.Agent):
    def __init__(self):
        core.Agent.__init__(self, OVERRIDES)
