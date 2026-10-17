"""Unit test: with every extra weight at 0, v3_value must equal F's base_fn exactly.

Also reports how far the shipped extras move the value, and times both functions.

    python version_3/tools/check_value.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_KIT = os.path.dirname(_ROOT)
sys.path.insert(0, _KIT)
sys.path.insert(0, os.path.join(_ROOT, "v3"))

import catan_kit  # noqa: F401
import random

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn

import v3_value

ZERO_EXTRAS = {k: 0.0 for k in v3_value.DEFAULT_EXTRAS}


def positions(n_games=6, n_positions=40):
    """Successor positions from random-agent games, so the features are varied."""
    out = []
    for seed in range(500000, 500000 + n_games):
        random.seed(seed)
        g = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=seed, vps_to_win=10)
        for step in range(600):
            if g.winning_color() is not None:
                break
            acts = g.playable_actions
            if len(acts) > 1 and step % 7 == 0:
                try:
                    c = g.copy()
                    c.execute(random.choice(acts))
                    out.append((c, Color.RED))
                    out.append((c, Color.BLUE))
                except Exception:
                    pass
            try:
                g.play_tick()
            except Exception:
                break
            if len(out) >= n_positions * n_games:
                break
    return out


def main():
    base = get_value_fn("base_fn", DEFAULT_WEIGHTS)
    ours = v3_value.make_value_fn(DEFAULT_WEIGHTS, ZERO_EXTRAS)
    ps = positions()
    worst = 0.0
    t_base = t_ours = 0.0
    for g, c in ps:
        t0 = time.process_time()
        a = base(g, c)
        t1 = time.process_time()
        b = ours(g, c)
        t2 = time.process_time()
        t_base += t1 - t0
        t_ours += t2 - t1
        worst = max(worst, abs(a - b) / max(1.0, abs(a)))
    print(f"positions           {len(ps)}")
    print(f"max relative diff   {worst:.3e}   (must be 0)")
    print(f"base_fn             {1000*t_base/max(1,len(ps)):.3f} ms/call")
    print(f"v3_value            {1000*t_ours/max(1,len(ps)):.3f} ms/call")
    assert worst == 0.0, "zero extras must reproduce F exactly"

    _conf = v3_value.load_config(os.path.join(_ROOT, "v3"))
    shipped = v3_value.make_value_fn(_conf["base"], _conf["extra"])
    diffs = []
    for g, c in ps[:20]:
        a = base(g, c)
        b = shipped(g, c)
        diffs.append(abs(a - b) / max(1.0, abs(a)))
    print(f"shipped extras move the value by up to {max(diffs):.4f} (relative)")
    print("OK")


if __name__ == "__main__":
    main()
