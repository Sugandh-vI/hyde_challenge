"""Play one agent against a pool of opponents and log every result.

Each opponent's games are split into chunks with non-overlapping board seeds and run in parallel
worker processes with ``concurrent.futures.ProcessPoolExecutor`` (NOT ``multiprocessing.Pool``:
its workers are daemonic and ``catan_kit.match`` has to spawn one OS process per agent).
Every worker calls ``catan_kit.evaluate.run`` -- the script we are scored with -- so the CPU
clock, the masked observations and the seat alternation are exactly the official ones.

    python version_3/tools/run_pool.py --agent version_3/v3/agent.py --games 400 --workers 7
    python version_3/tools/run_pool.py --agent version_3/v3_presets/p1/agent.py --games 40 --workers 2

Rows are appended to version_3/results.csv (my_team/results.csv columns + "opponent").
"""
import os
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TOOLS)
_KIT = os.path.dirname(_ROOT)
sys.path.insert(0, _KIT)

from catan_kit._patch import ensure_hash_seed

ensure_hash_seed()          # may re-exec with PYTHONHASHSEED=0

import argparse
import csv
import math
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

DEFAULT_POOL = [
    os.path.join(_KIT, "examples", "f_agent.py"),
    os.path.join(_KIT, "my_team", "agent_v2.py"),
    os.path.join(_KIT, "examples", "random_agent.py"),
    os.path.join(_ROOT, "opponents", "f_k3", "agent.py"),
    os.path.join(_ROOT, "opponents", "f_eps", "agent.py"),
]

HEADER = ["timestamp", "tag", "agent_a", "agent_b", "games", "seed0", "win_rate", "ci95",
          "mean_cpu_ms", "worst_cpu_ms", "games_over_budget", "errors", "timeouts", "crashes",
          "opponent"]


def _worker(job):
    agent_path, opp_path, games, seed0 = job
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from catan_kit.evaluate import run
    res = run(agent_path, opp_path, games, seed0=seed0, quiet=True)
    return res


def _chunks(games, workers):
    """Split into even-sized chunks (a board is always played twice, once per seat)."""
    pairs = games // 2
    rest = games % 2
    workers = max(1, min(workers, pairs if pairs else 1))
    base = pairs // workers
    extra = pairs % workers
    out = []
    start = 0
    for w in range(workers):
        p = base + (1 if w < extra else 0)
        if p == 0:
            continue
        g = 2 * p + (rest if w == workers - 1 else 0)
        out.append((g, start))
        start += 2 * p
    return out


def aggregate(results):
    games = sum(r["games"] for r in results)
    wins = sum(r["wins"] for r in results)
    rate = wins / games if games else 0.0
    cpu_weighted = sum(r["mean_cpu_s"] * r["games"] for r in results)
    return dict(
        games=games, wins=wins, win_rate=rate,
        ci95=1.96 * math.sqrt(max(rate * (1 - rate), 0.0) / games) if games else 0.0,
        mean_cpu_s=cpu_weighted / games if games else 0.0,
        worst_cpu_s=max((r["worst_cpu_s"] for r in results), default=0.0),
        games_over_budget=sum(r["games_over_budget"] for r in results),
        errors=sum(r["errors"] for r in results),
        timeouts=sum(r["timeouts"] for r in results),
        crashes=sum(r["crashes"] for r in results),
        decisions=sum(r.get("decisions", 0) for r in results),
    )


def run_match(agent, opp, games, seed0, workers):
    chunks = _chunks(games, workers)
    jobs = [(agent, opp, g, seed0 + start // 2) for g, start in chunks]
    if workers <= 1 or len(jobs) == 1:
        results = [_worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            results = list(pool.map(_worker, jobs))
    return aggregate(results)


def main():
    p = argparse.ArgumentParser(description="Play an agent against a pool of opponents.")
    p.add_argument("--agent", default=os.path.join(_ROOT, "v3", "agent.py"))
    p.add_argument("--pool", nargs="*", default=None, help="opponent agent paths")
    p.add_argument("--games", type=int, default=400, help="games against EACH opponent")
    p.add_argument("--seed0", type=int, default=500000)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--tag", default="pool")
    p.add_argument("--csv", default=os.path.join(_ROOT, "results.csv"))
    args = p.parse_args()

    agent = os.path.abspath(args.agent)
    pool = [os.path.abspath(x) for x in (args.pool or DEFAULT_POOL)]
    print(f"agent  {agent}")
    print(f"games  {args.games} per opponent, seed0={args.seed0}, workers={args.workers}\n")

    rows = []
    for opp in pool:
        t0 = time.time()
        res = run_match(agent, opp, args.games, args.seed0, args.workers)
        stem = os.path.splitext(os.path.basename(opp))[0]
        name = os.path.basename(os.path.dirname(opp)) if stem == "agent" else stem
        rows.append((name, opp, res))
        print(f"{name:<12} {res['win_rate']:.3f} +-{res['ci95']:.3f}  "
              f"({res['wins']}/{res['games']})  cpu {res['mean_cpu_s']*1000:6.1f} ms "
              f"worst {res['worst_cpu_s']*1000:6.1f}  over {res['games_over_budget']:3d}  "
              f"err {res['errors']:3d}  to {res['timeouts']} cr {res['crashes']}  "
              f"[{time.time()-t0:.0f}s]")

    stamp = datetime.now(timezone.utc).isoformat()
    new = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        for name, opp, res in rows:
            w.writerow([stamp, args.tag, agent, opp, res["games"], args.seed0,
                        f"{res['win_rate']:.4f}", f"{res['ci95']:.4f}",
                        f"{res['mean_cpu_s']*1000:.1f}", f"{res['worst_cpu_s']*1000:.1f}",
                        res["games_over_budget"], res["errors"], res["timeouts"],
                        res["crashes"], name])
    print(f"\nappended {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
