"""Write a self-contained preset folder under version_6/presets/.

A preset is one agent, complete: `agent.py` at the top (the entry point the kit loads), `core.py` (a
copy of `version_6/v6/agent.py`), every helper module, the value-net weights and the opening model,
all in the same folder.  Nothing is imported from outside it, so the folder can be copied alone:

    python -m catan_kit.evaluate --a version_6/presets/<name>/agent.py --b examples/f_agent.py --games 1000

Usage (from `catan-hackathon-agent/`):

    python version_6/tools/sync_preset.py              # write every preset in PRESETS
    python version_6/tools/sync_preset.py s3_vf        # just one

Each preset's switch values are the ones its screening step settled on; the numbers behind them are in
version_6/EXPERIMENTS.md.
"""
import os
import pprint
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
V6 = os.path.join(ROOT, "v6")
BASE = os.path.join(ROOT, "base")
OUT = os.path.join(ROOT, "presets")

#: name -> (source directory, switch overrides, one-line note for the reader)
PRESETS = {
    "s2_clock": (
        V6, {},
        "Step 2. version_6 defaults: the value net on every main turn with an adaptive CPU budget "
        "instead of SUB3's fixed NET_MIN_CPU = 0.25 s, the counter-opening search capped at 0.24 s, "
        "agent2's planner budget as instance settings.",
    ),
    "s3_vf": (
        V6, {"vf": {"buildable_nodes": 2000.0}},
        "Step 3. s2_clock with buildable_nodes 1000 -> 2000, the largest point estimate of the "
        "value-function weight screen (+0.6 pt, CI +-1.3: NOT significant, do not prefer it over "
        "s2_clock on strength).",
    ),
    "s4_net": (
        V6, {"net_on_knight": True},
        "Step 4. s2_clock with the net allowed to override the pre-roll knight rule (the only one of "
        "the three widened decisions that did not lose: -0.1 pt +-1.7).",
    ),
    "s5_win": (
        V6, {"winfind_vp": 8},
        "Step 5. s2_clock plus the depth-3 win finder from actual VP >= 8 (+0.3 pt +-0.3, but 3 of "
        "800 games ran out of clock: NOT recommended).",
    ),
}

README = """# preset `{name}`

{note}

Self-contained: `agent.py` is the entry point the kit loads, `core.py` is
`version_6/v6/agent.py`, and every helper module, the value-net weights and the opening model sit in
this folder.  Nothing is imported from outside it.

```bash
python -m catan_kit.evaluate --a version_6/presets/{name}/agent.py --b examples/f_agent.py --games 1000
```

The switch values, the 800-game paired screens behind them and the final harness numbers are in
`version_6/EXPERIMENTS.md` (and in the README's *Arena update 5*).
"""

AGENT = '''"""preset `{name}`

{note}

Self-contained: `core.py` is `version_6/v6/agent.py`, and every helper module, the value-net
weights and the opening model it uses sit in this folder.  This file only chooses the switches.

    python -m catan_kit.evaluate --a version_6/presets/{name}/agent.py --b examples/f_agent.py --games 1000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core

OVERRIDES = {overrides}


class Agent(core.Agent):
    def __init__(self):
        core.Agent.__init__(self, OVERRIDES)
'''


def write(name, src, overrides, note, quiet=False):
    dst = os.path.join(OUT, name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for fn in sorted(os.listdir(src)):
        if fn.startswith(".") or fn == "__pycache__" or fn == "README.md":
            continue
        s = os.path.join(src, fn)
        if not os.path.isfile(s) or not fn.endswith((".py", ".json", ".npz")):
            continue
        if fn == "agent.py":
            shutil.copy2(s, os.path.join(dst, "core.py"))
        elif fn != "README.md":
            shutil.copy2(s, os.path.join(dst, fn))
    with open(os.path.join(dst, "agent.py"), "w") as f:
        f.write(AGENT.format(name=name, note=note, overrides=pprint.pformat(overrides, width=96)))
    with open(os.path.join(dst, "README.md"), "w") as f:
        f.write(README.format(name=name, note=note))
    if not quiet:
        print(f"{name}: {len(os.listdir(dst))} files from {os.path.relpath(src, ROOT)}")


def main():
    names = sys.argv[1:] or list(PRESETS)
    for name in names:
        src, overrides, note = PRESETS[name]
        write(name, src, overrides, note)


if __name__ == "__main__":
    main()
