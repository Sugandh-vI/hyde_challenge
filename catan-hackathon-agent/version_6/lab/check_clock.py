"""Play one agent under a simulated harness clock and report what its decisions cost.

    python check_clock.py PATH/TO/agent.py [games] [seed0]

For every decision the harness charge is `min(0.300 + 0.0125*k, 1.5)` seconds after k decisions,
plus 2.5 ms for the JSON view decode it does outside `decide()`; here the agent's own CPU is measured
with `time.process_time()` around `decide()`, which is the part it controls.  Prints the per-game
cost, the per-decision distribution, and how often the agent would have run out of clock.

    --starve S   force `obs.cpu_left = S` at every decision (default: play the clock honestly).  Use
                 a tiny S (0.01) to check that the fallback path still answers fast and legally.

The agent is loaded with `_load.load_agent`, so an agent whose helpers clash with another agent's
(a2, feats, ...) is fine as long as only one is loaded per process.
"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fastmatch as fm  # noqa: F401  (thread pinning, KIT path, catan imports)
from _load import load_agent  # noqa: E402
from catan_kit.obs import Obs, masked_copy  # noqa: E402
from catanatron.game import Game  # noqa: E402
from catanatron.models.player import Color, Player, RandomPlayer  # noqa: E402

CPU_BASE, CPU_STEP, CPU_MAX = 0.300, 0.0125, 1.5
IO = 0.0025


class Seat(Player):
    def __init__(self, color, agent, seed):
        super().__init__(color)
        self.agent = agent
        self.rng = random.Random(seed)
        self.fallback = random.Random(seed + 1)
        self.costs = []
        self.cpu = 0.0
        self.calls = 0
        self.over = 0

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) == 1:
            return actions[0]
        allow = min(CPU_BASE + (self.calls + 1) * CPU_STEP, CPU_MAX)
        view = masked_copy(game, self.color, self.rng)
        view.state.players = [RandomPlayer(c) for c in view.state.colors]
        obs = Obs(view, self.color)
        obs.cpu_left = allow - self.cpu if STARVE is None else STARVE
        t0 = time.process_time()
        i = self.agent.decide(obs, actions)
        dt = time.process_time() - t0
        self.calls += 1
        self.cpu += dt + IO
        self.costs.append(dt)
        if self.cpu > allow:
            self.over += 1
        if not isinstance(i, int) or not 0 <= i < len(actions):
            self.costs[-1] = -1.0          # mark an illegal index
            return self.fallback.choice(actions)
        return actions[i]


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] * 1000


def main():
    path = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 500000
    global STARVE
    STARVE = None
    if "--starve" in sys.argv:
        STARVE = float(sys.argv[sys.argv.index("--starve") + 1])
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    Agent = load_agent(path, "checked_agent").Agent
    fmod = load_agent(os.path.join(HERE, "f_agent.py"), "checked_f")

    rows = []
    for k in range(games):
        seed = seed0 + k
        me = Seat(Color.RED, Agent(), seed)
        other = Seat(Color.BLUE, fmod.Agent(), seed + 7)
        for seat in (me, other):
            if seat is not me:
                seat.agent = fmod.Agent()          # F is not clocked; its Seat use is fine
                seat.costs = []
        random.seed(seed)
        game = Game(players=[me, other], seed=seed, vps_to_win=10)
        random.seed(seed)
        game.play()
        rows.append((me.calls, me.cpu, max(me.costs), sum(me.costs), me.over, list(me.costs)))
        print(f"game {seed}: {me.calls:3d} decisions, CPU {1000 * me.cpu:6.0f} ms "
              f"(agent {1000 * sum(me.costs):6.0f} + io {1000 * IO * me.calls:4.0f}), "
              f"worst {1000 * max(me.costs):6.1f} ms, over {me.over}")
    costs = sorted(c for r in rows for c in r[5])
    print(f"\nper-decision CPU: p50 {pct(costs, 0.50):6.2f} ms, p95 {pct(costs, 0.95):6.2f} ms, "
          f"p99 {pct(costs, 0.99):6.2f} ms, max {max(costs) * 1000:6.2f} ms, "
          f"n {len(costs)}")
    print(f"summary: mean CPU {1000 * sum(r[1] for r in rows) / len(rows):.0f} ms/game, "
          f"worst decision {1000 * max(r[2] for r in rows):.1f} ms, "
          f"games over budget {sum(1 for r in rows if r[4])}/{len(rows)}"
          + (f", forced cpu_left = {STARVE}" if STARVE is not None else ""))


if __name__ == "__main__":
    main()
