"""In-process paired match runner (no per-agent process, no CPU clock) for quick screening.

Uses the kit's masked_copy + Obs so agents see exactly what they see in the harness.
Each board seed is played twice with seats swapped and the same dice stream.
Usage: python fastmatch.py modA:ClassA modB:ClassB N_BOARDS [seed0] [procs]
"""
import importlib
import os
import random
import sys
import time

KIT = os.environ.get("KIT")
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import catan_kit  # noqa: F401  trade-order patch
from catan_kit.match import dice_seed
from catan_kit.obs import Obs, masked_copy
from catanatron.game import Game
from catanatron.models.player import Color, Player, RandomPlayer


def load(spec):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


class Seat(Player):
    def __init__(self, color, agent, seed):
        super().__init__(color)
        self.agent = agent
        self.mask_rng = random.Random(seed ^ 0x5EED)
        self.cpu = 0.0
        self.calls = 0
        self.worst = 0.0

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) == 1:
            return actions[0]
        view = masked_copy(game, self.color, self.mask_rng)
        view.state.players = [RandomPlayer(c) for c in view.state.colors]
        t0 = time.process_time()
        obs = Obs(view, self.color)
        obs.cpu_left = 1.0
        i = self.agent.decide(obs, actions)
        dt = time.process_time() - t0
        self.cpu += dt
        self.worst = max(self.worst, dt)
        self.calls += 1
        return actions[i]


def play_one(args):
    spec_a, spec_b, board_seed, a_first = args
    A, B = load(spec_a), load(spec_b)
    sa = Seat(Color.RED if a_first else Color.BLUE, A(), board_seed)
    sb = Seat(Color.BLUE if a_first else Color.RED, B(), board_seed + 1)
    players = [sa, sb] if a_first else [sb, sa]
    random.seed(board_seed)
    game = Game(players=players, seed=board_seed, vps_to_win=10)
    random.seed(dice_seed(board_seed, 0))      # same dice for both seat orders of a board
    winner = game.play()
    w = None if winner is None else ("a" if winner == sa.color else "b")
    return dict(seed=board_seed, a_first=a_first, winner=w, turns=game.state.num_turns,
                a_cpu=sa.cpu, a_calls=sa.calls, a_worst=sa.worst, b_cpu=sb.cpu, b_calls=sb.calls,
                a_st=stats(game, sa.color), b_st=stats(game, sb.color))


def stats(game, color):
    st = game.state
    i = st.color_to_index[color]
    ps = st.player_state
    buys = sum(1 for r in st.action_records
               if r.action.color == color and r.action.action_type.name == "BUY_DEVELOPMENT_CARD")
    return dict(vp=ps[f"P{i}_ACTUAL_VICTORY_POINTS"], sett=5 - ps[f"P{i}_SETTLEMENTS_AVAILABLE"],
                city=4 - ps[f"P{i}_CITIES_AVAILABLE"], lr=int(ps[f"P{i}_HAS_ROAD"]),
                la=int(ps[f"P{i}_HAS_ARMY"]), knights=ps[f"P{i}_PLAYED_KNIGHT"],
                vpcards=ps[f"P{i}_VICTORY_POINT_IN_HAND"], devbuys=buys)


def main():
    spec_a, spec_b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 500000
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else os.cpu_count()
    jobs = [(spec_a, spec_b, seed0 + i, af) for i in range(n) for af in (True, False)]
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(procs) as p:
        res = p.map(play_one, jobs, chunksize=4)
    wall = time.time() - t0
    g = len(res)
    wins = sum(r["winner"] == "a" for r in res)
    first = [r for r in res if r["a_first"]]
    second = [r for r in res if not r["a_first"]]
    pa = wins / g
    ci = 1.96 * (pa * (1 - pa) / g) ** 0.5
    # paired (per-board) win rate: board score = mean of its two games
    boards = {}
    for r in res:
        boards.setdefault(r["seed"], []).append(r["winner"] == "a")
    import statistics
    bs = [sum(v) / 2 for v in boards.values()]
    pci = 1.96 * statistics.pstdev(bs) / len(bs) ** 0.5
    print(f"{spec_a} vs {spec_b}: {g} games, A win {pa:.3f} +-{ci:.3f} (paired +-{pci:.3f}) "
          f"| A first {sum(r['winner']=='a' for r in first)/len(first):.3f} "
          f"second {sum(r['winner']=='a' for r in second)/len(second):.3f} "
          f"| no-winner {sum(r['winner'] is None for r in res)} "
          f"| A cpu/game {1000*sum(r['a_cpu'] for r in res)/g:.0f} ms, max {1000*max(r['a_cpu'] for r in res):.0f} ms, "
          f"worst decision {1000*max(r['a_worst'] for r in res):.0f} ms, dec/game {sum(r['a_calls'] for r in res)/g:.0f} "
          f"| B cpu/game {1000*sum(r['b_cpu'] for r in res)/g:.0f} ms "
          f"| turns med {statistics.median(r['turns'] for r in res)} | wall {wall:.0f}s")
    keys = ["vp", "sett", "city", "lr", "la", "knights", "vpcards", "devbuys"]
    for side in ("a", "b"):
        for label, sub in (("won", [r for r in res if r["winner"] == side]),
                           ("lost", [r for r in res if r["winner"] not in (side, None)])):
            if sub:
                avg = {k: sum(r[side + "_st"][k] for r in sub) / len(sub) for k in keys}
                print(f"  {side.upper()} when {label} ({len(sub)}): " + " ".join(f"{k}={v:.2f}" for k, v in avg.items()))
    import json
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl"), "a") as f:
        f.write(json.dumps(dict(t=time.strftime("%H:%M"), a=spec_a, b=spec_b, seed0=seed0, games=g,
                                win=round(pa, 4), ci=round(ci, 4), pci=round(pci, 4),
                                a_cpu_ms=round(1000 * sum(r["a_cpu"] for r in res) / g))) + chr(10))


if __name__ == "__main__":
    main()
