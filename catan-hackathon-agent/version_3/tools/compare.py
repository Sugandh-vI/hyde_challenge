"""Paired comparison of two agents against the same opponent on the SAME boards.

Both agents play the identical list of (board seed, game index) pairs, once each, through
``catan_kit.match.play_chunk``.  Because the dice and the boards are identical game by game, the
difference in win rate can be tested per game: the paired 95% interval is usually much tighter
than the interval of either win rate alone, which is what makes a 40-game comparison useful.

    python version_3/tools/compare.py --a version_3/v3_presets/p3/agent.py \
                                      --b version_3/v3_presets/p2/agent.py \
                                      --opponent examples/f_agent.py --games 400
"""
import os
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TOOLS)
_KIT = os.path.dirname(_ROOT)
sys.path.insert(0, _KIT)

from catan_kit._patch import ensure_hash_seed

ensure_hash_seed()

import argparse
import math
import statistics
import time


def play(path_a, path_opp, games, seed0):
    from catan_kit.match import play_chunk
    seeds = [seed0 + i // 2 for i in range(games)]
    records, info = play_chunk(path_a, path_opp, seeds, list(range(games)))
    return records, info


def cells(records):
    return [1 if r["winner"] == "a" else 0 for r in records]


def main():
    p = argparse.ArgumentParser(description="Paired comparison of two agents on the same seeds.")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--opponent", default=os.path.join(_KIT, "examples", "f_agent.py"))
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--seed0", type=int, default=500000)
    args = p.parse_args()
    n = args.games

    print(f"opponent {args.opponent}\ngames {n}, seed0 {args.seed0}, seats alternated\n")
    t0 = time.time()
    rec_a, info_a = play(args.a, args.opponent, n, args.seed0)
    print(f"A {args.a}: {time.time()-t0:.0f}s, load {info_a['a_init_s']:.2f}s")
    t0 = time.time()
    rec_b, info_b = play(args.b, args.opponent, n, args.seed0)
    print(f"B {args.b}: {time.time()-t0:.0f}s, load {info_b['a_init_s']:.2f}s\n")

    a, b = cells(rec_a), cells(rec_b)
    wa, wb = sum(a) / n, sum(b) / n
    d = [x - y for x, y in zip(a, b)]
    mean_d = sum(d) / n
    if n > 1:
        sd = statistics.stdev(d)
        ci = 1.96 * sd / math.sqrt(n)
    else:
        ci = 0.0
    only_a = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    only_b = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)

    # In each match argument A is side "a", so B's numbers come from rec_b["a"] (not "b": that is
    # the shared opponent, whose CPU we do not care about here).
    cpu_a = [r["a"]["cpu_s"] for r in rec_a]
    cpu_b = [r["a"]["cpu_s"] for r in rec_b]
    print(f"A win rate  {wa:.4f}  ({sum(a)}/{n})")
    print(f"B win rate  {wb:.4f}  ({sum(b)}/{n})")
    print(f"A - B       {mean_d:+.4f} +-{ci:.4f}   (paired 95% interval)")
    print(f"discordant  A only {only_a}, B only {only_b}")
    print(f"A cpu       {statistics.mean(cpu_a)*1000:.1f} ms/game, worst {max(cpu_a)*1000:.0f} ms, "
          f"over {sum(1 for r in rec_a if r['a']['over'])}")
    print(f"B cpu       {statistics.mean(cpu_b)*1000:.1f} ms/game, worst {max(cpu_b)*1000:.0f} ms, "
          f"over {sum(1 for r in rec_b if r['a']['over'])}")
    print(f"A errors    {sum(r['a']['errors'] for r in rec_a)}, "
          f"timeouts {sum(r['a']['timeouts'] for r in rec_a)}, "
          f"crashes {sum(r['a']['crashes'] for r in rec_a)}")
    print(f"B errors    {sum(r['a']['errors'] for r in rec_b)}, "
          f"timeouts {sum(r['a']['timeouts'] for r in rec_b)}, "
          f"crashes {sum(r['a']['crashes'] for r in rec_b)}")


if __name__ == "__main__":
    main()
