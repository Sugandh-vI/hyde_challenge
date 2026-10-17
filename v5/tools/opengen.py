"""Generate opening training data: an explorer (F mid-game, softmax-sampled opening) vs the real F.

Each row: features of (explorer side, F side, explorer-first flag) at the end of setup, and whether
the explorer won. Usage: python opengen.py OUT.npz N_GAMES SEED0 PROCS
"""
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("KIT", ""))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import catan_kit  # noqa: F401
from catan_kit.match import dice_seed
from catanatron.game import Game
from catanatron.models.enums import ActionType, SETTLEMENT, CITY
from catanatron.models.player import Color, Player
from catanatron.players.value import DEFAULT_WEIGHTS, base_fn

from openfeat import game_features, node_pips
from open_agent import spot_score, _neighbors
from vf import make_vf
VF = make_vf(dict(lr_always=2e3, port2=3e7, port3=1e6))  # explorer mid-game = current best

FVF = base_fn(DEFAULT_WEIGHTS)


def f_choice(game, actions, color, vf=None):
    vf = vf or FVF
    best, bv = actions[0], None
    for a in actions:
        g = game.copy()
        g.execute(a)
        v = vf(g, color)
        if bv is None or v > bv:
            best, bv = a, v
    return best


class FPlayer(Player):
    def decide(self, game, actions):
        return actions[0] if len(actions) == 1 else f_choice(game, actions, self.color)


class Explorer(Player):
    def __init__(self, color, rng, temp, mode):
        super().__init__(color)
        self.rng, self.temp, self.mode = rng, temp, mode

    def decide(self, game, actions):
        if len(actions) == 1:
            return actions[0]
        st = game.state
        if st.is_initial_build_phase and actions[0].action_type == ActionType.BUILD_SETTLEMENT:
            b = st.buildings_by_color[self.color]
            mine = b[SETTLEMENT] + b[CITY]
            if self.mode == "counter" and self.rng.random() > 0.25:
                import opening
                opp = next(c for c in st.colors if c != self.color)
                node = opening.choose(st.board, [a.value for a in actions], mine,
                                      st.buildings_by_color[opp][SETTLEMENT], st.colors[0] == self.color)
                return next(a for a in actions if a.value == node)
            if self.mode == "pips":
                sc = [sum(node_pips(st.board, a.value).values()) for a in actions]
            else:
                sc = [spot_score(st.board, a.value, mine) for a in actions]
            top = sorted(range(len(actions)), key=lambda i: -sc[i])[:10]
            m = max(sc[i] for i in top)
            w = [math.exp((sc[i] - m) / self.temp) for i in top]
            return actions[self.rng.choices(top, weights=w)[0]]
        if st.is_initial_build_phase and actions[0].action_type == ActionType.BUILD_ROAD:
            b = st.buildings_by_color[self.color]
            mine = b[SETTLEMENT] + b[CITY]
            open_nodes = set(st.board.buildable_node_ids(self.color, initial_build_phase=True))

            def rs(a):
                u, v = a.value
                far = v if u in mine else u
                return max((spot_score(st.board, n, mine) for n in _neighbors(st.board, far)
                            if n in open_nodes), default=0.0)
            return max(actions, key=rs)
        return f_choice(game, actions, self.color, VF)


MODES = os.environ.get("MODES", "score,score,pips").split(",")


def one(seed):
    rng = random.Random(seed * 7 + 1)
    temp = rng.choice([0.5, 1.0, 2.0, 4.0])
    mode = rng.choice(MODES)
    x_first = seed % 2 == 0
    cx, cf = (Color.RED, Color.BLUE) if x_first else (Color.BLUE, Color.RED)
    x, f = Explorer(cx, rng, temp, mode), FPlayer(cf)
    random.seed(seed)
    game = Game(players=[x, f] if x_first else [f, x], seed=seed, vps_to_win=10)
    random.seed(dice_seed(seed, 0))
    while game.state.is_initial_build_phase:
        game.play_tick()
    b = game.state.buildings_by_color
    xs, fs = b[cx][SETTLEMENT], b[cf][SETTLEMENT]
    feats = game_features(game.state.board, xs, fs, x_first)
    w = game.play()
    return feats, (1 if w == cx else 0), seed


def main():
    out, n, seed0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(procs) as p:
        res = p.map(one, range(seed0, seed0 + n), chunksize=8)
    X = np.array([r[0] for r in res], dtype=np.float32)
    y = np.array([r[1] for r in res], dtype=np.int8)
    s = np.array([r[2] for r in res])
    np.savez(out, X=X, y=y, seeds=s)
    print(f"{n} games in {time.time()-t0:.0f}s, explorer win {y.mean():.3f} -> {out}")


if __name__ == "__main__":
    main()
