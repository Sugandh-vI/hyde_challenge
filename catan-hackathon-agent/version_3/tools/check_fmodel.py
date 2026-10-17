"""Diagnostic: how often does the opening's F-model predict F's actual setup pick?

F is catanatron's ValueFunctionPlayer, which is 1-ply argmax of base_fn over game.playable_actions
in engine order.  Our model does the same but may restrict the candidate set to the top
``opp_max_eval`` nodes by a pip proxy.  This script measures the agreement for several caps.

    python version_3/tools/check_fmodel.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.dirname(_ROOT))
sys.path.insert(0, os.path.join(_ROOT, "v3"))

import catan_kit  # noqa: F401
import random

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.players.value import DEFAULT_WEIGHTS, ValueFunctionPlayer, get_value_fn

import v3_opening

BASE = get_value_fn("base_fn", DEFAULT_WEIGHTS)
CAPS = [None, 12, 10, 8, 6]


class RecordingF(ValueFunctionPlayer):
    """F, plus recording of (our model's prediction) for every setup settlement it places."""

    def __init__(self, color, record):
        super().__init__(color)
        self.record = record

    def decide(self, game, playable_actions):
        action = super().decide(game, playable_actions)
        if playable_actions and playable_actions[0].action_type.name == "BUILD_SETTLEMENT":
            row = {"n": len(playable_actions), "actual": action.value, "preds": {}}
            for cap in CAPS:
                cfg = v3_opening.cfg_from({"opp_max_eval": cap if cap is not None else 0})
                row["preds"][cap] = v3_opening._f_pick(game, self.color, cfg, BASE)
            self.record.append(row)
        return action


def main():
    rec = []
    for seed in range(500000, 500024):
        random.seed(seed)
        g = Game([RecordingF(Color.RED, rec), RecordingF(Color.BLUE, rec)], seed=seed, vps_to_win=10)
        random.seed(seed * 7 + 3)
        g.play()
    print(f"setup settlement decisions recorded: {len(rec)}")
    for cap in CAPS:
        ok = sum(1 for r in rec if r["preds"][cap] == r["actual"])
        print(f"  opp_max_eval={str(cap):>4}: exact agreement {ok}/{len(rec)} = {ok/len(rec):.3f}")
    worst = [r for r in rec if r["preds"][10] != r["actual"]][:5]
    for r in worst:
        print("   miss:", r["n"], "actual", r["actual"], "pred", r["preds"][10])


if __name__ == "__main__":
    main()
