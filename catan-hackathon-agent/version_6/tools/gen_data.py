"""Generate positions + outcomes for retraining the value net (Step 7 of version_6).

Plays games in-process, records `feats.features(game, color)` at every real decision for both seats,
and labels each row with that seat's game result.  Positions come from the strongest agent we have
(`version_6/presets/<best>/agent.py`) against a mix of opponents, so the data covers the states the
net is actually asked about.

    python version_6/tools/gen_data.py --games 2000 --workers 2 --out version_6/data

Writes one `shard_XXXX.npz` per chunk (`X` float32, `y` int8, `seed`, `color`, `turn`), so a long run
can be stopped and the finished shards used as they are.  Seeds for training data start at 100000 by
default; keep them disjoint from the 500000+ screening seeds and the 990000+ final seeds.

Split for training: by game seed (`train_net.py --val-mod 8`), never by row, so both seats of a game
land on the same side of the split.
"""
import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.dirname(HERE)
KIT = os.environ.get("KIT") or os.path.dirname(V6)
LAB = os.path.join(V6, "lab")
sys.path.insert(0, LAB)
sys.path.insert(0, KIT)

MATCHUPS = ("self", "self", "f", "base")       # repeated entries = weights


def _pin():
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def run_chunk(job):
    """(agent_path, opponent_key, seeds, out_path, rows_per_color) -> (games, rows)."""
    agent_path, opp_key, seeds, out_path, per_color = job
    _pin()
    import numpy as np
    from catan_kit.obs import Obs, masked_copy
    from catanatron.game import Game
    from catanatron.models.player import Color, Player, RandomPlayer
    from _load import load_agent

    class Recorder(Player):
        """Plays `agent` and records the feature row of the seat it plays, from the real game."""

        def __init__(self, color, agent, feats, seed):
            super().__init__(color)
            self.agent = agent
            self.feats = feats
            self.rng = random.Random(seed)
            self.rows = []
            self.calls = 0
            self.cap = per_color

        def decide(self, game, playable_actions):
            actions = list(playable_actions)
            if len(actions) > 1:
                self.rows.append(self.feats.features(game, self.color))
                self.calls += 1
            view = masked_copy(game, self.color, self.rng)
            view.state.players = [RandomPlayer(c) for c in view.state.colors]   # no opponent object
            obs = Obs(view, self.color)
            obs.cpu_left = min(0.300 + 0.0125 * self.calls, 1.5)
            return actions[self.agent.decide(obs, actions)]

    mod = load_agent(agent_path, "gen_agent")
    feats = load_agent(os.path.join(os.path.dirname(os.path.abspath(agent_path)), "feats.py"),
                       "gen_feats")
    if opp_key == "self":
        opp_mod = mod
    elif opp_key == "base":
        opp_mod = load_agent(os.path.join(V6, "base", "agent.py"), "gen_base")
    else:
        opp_mod = load_agent(os.path.join(LAB, "f_agent.py"), "gen_f")

    X, Y, S, C, T = [], [], [], [], []
    games = 0
    for seed in seeds:
        me = Recorder(Color.RED, mod.Agent(), feats, seed)
        other = Recorder(Color.BLUE, opp_mod.Agent(), feats, seed + 1)
        random.seed(seed)
        g = Game(players=[me, other], seed=seed, vps_to_win=10)
        random.seed(seed)
        winner = g.play()
        for rec in (me, other):
            rows = rec.rows
            if len(rows) > rec.cap:                    # evenly spaced subsample, keep the last
                step = len(rows) / float(rec.cap)
                rows = [rows[min(len(rows) - 1, int(i * step))] for i in range(rec.cap)]
            for r in rows:
                X.append(r)
                Y.append(1 if winner == rec.color else 0)
                S.append(seed)
                C.append(0 if rec.color == Color.RED else 1)
                T.append(g.state.num_turns)
        games += 1
    X = np.asarray(X, dtype=np.float32)
    np.savez_compressed(out_path, X=X, y=np.asarray(Y, dtype=np.int8),
                        seed=np.asarray(S, dtype=np.int64), color=np.asarray(C, dtype=np.int8),
                        turn=np.asarray(T, dtype=np.int32))
    return games, len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=os.path.join(V6, "presets", "s2_clock", "agent.py"))
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=100000)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--rows-per-color", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(V6, "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from catan_kit._patch import ensure_hash_seed
    ensure_hash_seed()

    seeds = list(range(args.seed0, args.seed0 + args.games))
    jobs = []
    for i in range(0, len(seeds), args.chunk):
        chunk = seeds[i:i + args.chunk]
        opp = MATCHUPS[(i // args.chunk) % len(MATCHUPS)]
        jobs.append((args.agent, opp, chunk, os.path.join(args.out, "shard_%04d.npz" % (i // args.chunk)),
                     args.rows_per_color))
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    done_g = done_r = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, (g, r) in enumerate(ex.map(run_chunk, jobs)):
            done_g += g
            done_r += r
            print(f"shard {k + 1}/{len(jobs)}: +{g} games, +{r} rows "
                  f"(total {done_g} games, {done_r} rows, {time.time() - t0:.0f}s)", flush=True)
    print(f"done: {done_g} games, {done_r} rows in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
