"""In-process paired match runner (no per-agent process) for quick screening.

Two modes:

  lab (default)   obs.cpu_left = 1.0 at every decision, exactly as before: no clock pressure, so a
                  candidate shows its strength with unlimited thinking time.
  --clock         the harness clock, simulated: allowance = min(0.300 + 0.0125*k, 1.5) after k
                  decisions, cpu_left = allowance - CPU this agent has used in this game.  If the
                  agent's CPU passes the allowance, its answer is discarded and uniformly random
                  legal moves are played for the rest of the game, exactly like catan_kit.match.
                  The harness charges the agent *process*, so every decision also carries the JSON
                  decode of the view and Obs construction outside decide(); --io_ms charges that
                  (default 2.5 ms, measured: harness 299 ms/game vs 172 ms of in-decide CPU).
                  --cpu_scale S charges S x the measured in-decide CPU instead (emulating a slower
                  machine); keep it at 1.0 for agents that time their own search, because their
                  deadlines and the charged time then disagree.

Instrumentation (read by the lab, never printed by the agent): for agents with the SUB3 layout
(`_main` + `net.scores`) it counts main-turn decisions, net consultations and overrides.

Usage:
  python fastmatch.py modA:ClassA modB:ClassB BOARDS [seed0] [procs] [--clock] [--cpu_scale S]
"""
import importlib
import os
import random
import sys
import time

# The harness runs each agent in its own process on one core with OMP/MKL threads pinned to 1
# (catan_kit/host.py).  Without that, numpy's OpenBLAS spawns one thread per core and the measured
# CPU of a single decision triples (measured: 97 ms/game pinned vs 345 ms/game unpinned), which is
# neither what the harness charges nor how the evaluation machine looks.  Pin before numpy loads.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

KIT = os.environ.get("KIT", "")
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import catan_kit  # noqa: F401  trade-order patch
from catan_kit.match import CPU_BASE_S, CPU_MAX_S, CPU_PER_DECISION_S, dice_seed
from catan_kit.obs import Obs, masked_copy
from catanatron.game import Game
from catanatron.models.player import Color, Player, RandomPlayer


def load(spec):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


def instrument(agent):
    """Count main decisions, net consultations and overrides.  Works for SUB3-shaped agents."""
    stats = dict(main=0, net=0, override=0, enabled=False)
    try:
        net = getattr(agent, "net", None)
        orig_main = agent._main
        orig_scores = getattr(net, "scores", None)
        if orig_scores is None:
            return stats
    except Exception:
        return stats
    stats["enabled"] = True
    box = {"argmax": None, "before": 0}

    def scores_wrap(obs, actions, **kw):
        vals = orig_scores(obs, actions, **kw)
        try:
            box["argmax"] = int(vals.argmax())
            stats["net"] += 1
        except Exception:
            pass
        return vals

    def main_wrap(world, me, actions, deadline):
        box["argmax"] = None
        before = stats["net"]
        i = orig_main(world, me, actions, deadline)
        stats["main"] += 1
        if stats["net"] > before and box["argmax"] is not None and box["argmax"] != i:
            stats["override"] += 1
        return i

    try:
        net.scores = scores_wrap
        agent._main = main_wrap
    except Exception:
        stats["enabled"] = False
    return stats


class Seat(Player):
    def __init__(self, color, agent, seed, clock=False, cpu_scale=1.0, io=0.0025):
        super().__init__(color)
        self.agent = agent
        self.io = io
        self.mask_rng = random.Random(seed ^ 0x5EED)
        self.fallback = random.Random(seed)
        self.clock = clock
        self.cpu_scale = cpu_scale
        self.cpu = 0.0
        self.calls = 0
        self.worst = 0.0
        self.over = None
        self.random_moves = 0
        self.stats = instrument(agent)

    def allowance(self):
        return min(CPU_BASE_S + (self.calls + 1) * CPU_PER_DECISION_S, CPU_MAX_S)

    def _random(self, actions):
        self.random_moves += 1
        return self.fallback.choice(actions)

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) == 1:
            return actions[0]
        if self.clock and self.over:
            return self._random(actions)
        allow = self.allowance()
        view = masked_copy(game, self.color, self.mask_rng)
        view.state.players = [RandomPlayer(c) for c in view.state.colors]
        obs = Obs(view, self.color)
        obs.cpu_left = (allow - self.cpu) if self.clock else 1.0
        t0 = time.process_time()
        i = self.agent.decide(obs, actions)
        dt = time.process_time() - t0
        charged = dt * self.cpu_scale + self.io
        self.cpu += charged
        self.calls += 1
        self.worst = max(self.worst, charged)
        if self.clock and self.cpu > allow:
            self.over = "cpu"
            return self._random(actions)       # the harness discards a late answer
        return actions[i]


def play_one(args):
    spec_a, spec_b, board_seed, a_first, opts = args
    opts = opts or {}
    A, B = load(spec_a), load(spec_b)
    sa = Seat(Color.RED if a_first else Color.BLUE, A(), board_seed, **opts)
    sb = Seat(Color.BLUE if a_first else Color.RED, B(), board_seed + 1, clock=False)
    players = [sa, sb] if a_first else [sb, sa]
    random.seed(board_seed)
    game = Game(players=players, seed=board_seed, vps_to_win=10)
    random.seed(dice_seed(board_seed, 0))      # same dice for both seat orders of a board
    winner = game.play()
    w = None if winner is None else ("a" if winner == sa.color else "b")
    return dict(seed=board_seed, a_first=a_first, winner=w, turns=game.state.num_turns,
                a_cpu=sa.cpu, a_calls=sa.calls, a_worst=sa.worst, b_cpu=sb.cpu, b_calls=sb.calls,
                over=1 if sa.over else 0, rnd=sa.random_moves,
                main=sa.stats["main"], net=sa.stats["net"], override=sa.stats["override"],
                wf=agent_stat(sa.agent, "winfind"), wfh=agent_stat(sa.agent, "winhit"),
                a_st=stats(game, sa.color), b_st=stats(game, sb.color))


def agent_stat(agent, name):
    st = getattr(agent, "stats", None)
    return st.get(name, 0) if isinstance(st, dict) else 0


def stats(game, color):
    st = game.state
    i = st.color_to_index[color]
    ps = st.player_state
    buys = sum(1 for r in st.action_records
               if r.action.color == color and r.action.action_type.name == "BUY_DEVELOPMENT_CARD")
    return dict(vp=ps[f"P{i}_ACTUAL_VICTORY_POINTS"], sett=5 - ps[f"P{i}_SETTLEMENTS_AVAILABLE"],
                city=4 - ps[f"P{i}_CITIES_AVAILABLE"], lr=int(ps[f"P{i}_HAS_ROAD"]),
                la=int(ps[f"P{i}_HAS_ARMY"]), knights=ps[f"P{i}_PLAYED_KNIGHTS"]
                if f"P{i}_PLAYED_KNIGHTS" in ps else ps.get(f"P{i}_PLAYED_KNIGHT", 0),
                vpcards=ps[f"P{i}_VICTORY_POINT_IN_HAND"], devbuys=buys)


def summarise(res, spec_a):
    import statistics
    g = len(res)
    wins = sum(r["winner"] == "a" for r in res)
    boards = {}
    for r in res:
        boards.setdefault(r["seed"], []).append(r["winner"] == "a")
    bs = [sum(v) / len(v) for v in boards.values()]
    pci = 1.96 * statistics.pstdev(bs) / len(bs) ** 0.5
    pa = wins / g
    ci = 1.96 * (pa * (1 - pa) / g) ** 0.5
    main = sum(r["main"] for r in res)
    net = sum(r["net"] for r in res)
    return dict(games=g, win=pa, ci=ci, pci=pci, boards=len(boards),
                cpu=1000 * sum(r["a_cpu"] for r in res) / g,
                worst=1000 * max(r["a_cpu"] for r in res),
                worst_dec=1000 * max(r["a_worst"] for r in res),
                dec=sum(r["a_calls"] for r in res) / g,
                over=sum(r["over"] for r in res), rnd=sum(r["rnd"] for r in res),
                main=main, net=net, override=sum(r["override"] for r in res),
                wf=sum(r["wf"] for r in res), wfh=sum(r["wfh"] for r in res),
                net_frac=(net / main if main else float("nan")))


def main():
    argv = [a for a in sys.argv[1:]]
    flags = dict(clock=False, cpu_scale=1.0, io=0.0025)
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--clock":
            flags["clock"] = True
            i += 1
        elif argv[i] == "--cpu_scale":
            flags["cpu_scale"] = float(argv[i + 1])
            i += 2
        elif argv[i] == "--io_ms":
            flags["io"] = float(argv[i + 1]) / 1000.0
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    spec_a, spec_b, n = rest[0], rest[1], int(rest[2])
    seed0 = int(rest[3]) if len(rest) > 3 else 500000
    procs = int(rest[4]) if len(rest) > 4 else os.cpu_count()
    jobs = [(spec_a, spec_b, seed0 + i, af, flags) for i in range(n) for af in (True, False)]
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(procs) as p:
        res = p.map(play_one, jobs, chunksize=4)
    s = summarise(res, spec_a)
    print(f"{spec_a} vs {spec_b}: {s['games']} games, A win {s['win']:.3f} +-{s['ci']:.3f} "
          f"(paired +-{s['pci']:.3f}) | cpu/game {s['cpu']:.0f} ms, max {s['worst']:.0f} ms, "
          f"worst decision {s['worst_dec']:.0f} ms, dec/game {s['dec']:.0f} "
          f"| net {s['net']}/{s['main']} main ({s['net_frac']:.2f}), overrides {s['override']} "
          f"| over budget {s['over']} ({s['rnd']} random moves)"
          + (f" | winfind {s['wf']} ({s['wfh']} hits)" if s["wf"] else "")
          + f" | wall {time.time()-t0:.0f}s")
    import json
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl"), "a") as f:
        f.write(json.dumps(dict(t=time.strftime("%H:%M"), a=spec_a, b=spec_b, seed0=seed0,
                                games=s["games"], win=round(s["win"], 4), ci=round(s["ci"], 4),
                                pci=round(s["pci"], 4), a_cpu_ms=round(s["cpu"]),
                                clock=flags["clock"], cpu_scale=flags["cpu_scale"],
                                net_frac=round(s["net_frac"], 3) if s["main"] else None,
                                over=s["over"])) + "\n")


if __name__ == "__main__":
    main()
