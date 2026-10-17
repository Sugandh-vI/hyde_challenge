"""Random search over version_3/v3/weights.json, scored by win rate against F.

Each candidate is a full copy of the agent in a temp folder with its own weights.json, evaluated
on ``--games`` games with seeds ``--seed0`` and up, in parallel worker processes.  The best
candidate seen is written to ``version_3/v3/weights_best.json`` (which the agent loads in
preference to weights.json), but only if it beats the best fitness stored so far, so a plain
``python version_3/tools/sweep.py`` can be run repeatedly and keeps improving.

    python version_3/tools/sweep.py --smoke                       # 3 candidates, 8 games, 2 workers
    python version_3/tools/sweep.py --pop 24 --gens 6 --games 200 --workers 7

Tuning seeds are 100000+ so they never overlap the development seeds (500000+) or the final
checks (990000+).
"""
import os
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TOOLS)
_KIT = os.path.dirname(_ROOT)
_V3 = os.path.join(_ROOT, "v3")
sys.path.insert(0, _KIT)

from catan_kit._patch import ensure_hash_seed

ensure_hash_seed()

import argparse
import copy
import csv
import json
import math
import os as _os
import random
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

#: extras sampled multiplicatively around their current value
EXTRA_KEYS = ["lr_always", "port2", "port3"]
#: extras that default to 0 and are sampled from a small signed grid
ZERO_EXTRA_GRIDS = {
    "opp_vp": [-3e13, -1e13, -3e12, 0.0, 3e12],
    "dist_city": [-2e6, -5e5, 0.0, 5e5, 2e6],
    "dist_settlement": [-1e6, -2.5e5, 0.0, 2.5e5, 1e6],
    "dist_dev": [-1e6, -2.5e5, 0.0, 2.5e5, 1e6],
}
#: indices in the 21-entry opening coefficient vector that the sweep may move
OPEN_KEYS = [6, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]   # n_res, ports, balances, open spots


def _candidate(rng, base, sigma=0.35, sigma_open=0.30):
    c = copy.deepcopy(base)
    for k in EXTRA_KEYS:
        v = float(c["extra"].get(k, 0.0))
        c["extra"][k] = v * math.exp(rng.gauss(0.0, sigma)) if v else rng.choice([1e3, 1e4])
    for k, grid in ZERO_EXTRA_GRIDS.items():
        c["extra"][k] = float(rng.choice(grid))
    coef = list(c["opening"]["coef"])
    for j in OPEN_KEYS:
        if j < len(coef):
            coef[j] = max(0.0, coef[j] * math.exp(rng.gauss(0.0, sigma_open)))
    c["opening"]["coef"] = coef
    c["opening"]["k1"] = base["opening"].get("k1", 6)
    c["opening"]["k2"] = base["opening"].get("k2", 10)
    return c


def _make_dir(config):
    d = tempfile.mkdtemp(prefix="v3-sweep-")
    for name in _os.listdir(_V3):
        if name.endswith(".py"):
            shutil.copyfile(_os.path.join(_V3, name), _os.path.join(d, name))
    with open(_os.path.join(d, "weights.json"), "w") as f:
        json.dump(config, f)
    return d


def _worker(job):
    d, games, seed0 = job
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from catan_kit.evaluate import run
    try:
        res = run(_os.path.join(d, "agent.py"), _os.path.join(_KIT, "examples", "f_agent.py"),
                  games, seed0=seed0, quiet=True)
        return float(res["win_rate"]), int(res["errors"]), int(res["games_over_budget"])
    except Exception:
        return 0.0, games, 0


def main():
    p = argparse.ArgumentParser(description="Random search over v3 weights, scored vs F.")
    p.add_argument("--pop", type=int, default=24, help="candidates per generation")
    p.add_argument("--gens", type=int, default=6, help="generations")
    p.add_argument("--games", type=int, default=200, help="games per candidate")
    p.add_argument("--seed0", type=int, default=100000, help="first tuning seed (100000+)")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--sigma-open", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=12345, help="seed of the search itself")
    p.add_argument("--best", default=os.path.join(_V3, "weights_best.json"))
    p.add_argument("--state", default=os.path.join(_TOOLS, "sweep_state.json"))
    p.add_argument("--log", default=os.path.join(_ROOT, "sweep_log.csv"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.pop, args.games, args.workers, args.gens = 3, 8, 2, 1
        args.best = os.path.join(_V3, "weights_smoke.json")
        args.state = os.path.join(_TOOLS, "sweep_state_smoke.json")
        args.log = os.path.join(_ROOT, "sweep_log_smoke.csv")
        print("[sweep] SMOKE: 3 candidates x 8 games x 1 generation, 2 workers")

    with open(os.path.join(_V3, "weights.json")) as f:
        base = json.load(f)
    best_fit = -1.0
    best_cfg = None
    if os.path.exists(args.state):
        try:
            with open(args.state) as f:
                st = json.load(f)
            best_fit, best_cfg = st["fitness"], st["config"]
            print(f"[sweep] resuming: stored best fitness {best_fit:.3f}")
        except Exception:
            pass
    if best_cfg is None:
        best_cfg = base

    rng = random.Random(args.seed)
    new_log = not os.path.exists(args.log) or os.path.getsize(args.log) == 0
    print(f"[sweep] {args.gens} generations x {args.pop} candidates x {args.games} games "
          f"vs F, seed0 {args.seed0}, {args.workers} workers")

    for gen in range(args.gens):
        t0 = time.time()
        cfgs = [_candidate(rng, base, args.sigma, args.sigma_open) for _ in range(args.pop)]
        dirs = [_make_dir(c) for c in cfgs]
        jobs = [(d, args.games, args.seed0 + gen * args.games) for d in dirs]
        try:
            with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
                out = list(pool.map(_worker, jobs))
        except Exception as e:
            print(f"[sweep] pool failed ({e}); running serially", file=sys.stderr)
            out = [_worker(j) for j in jobs]
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)

        fits = [o[0] for o in out]
        errs = sum(o[1] for o in out)
        over = sum(o[2] for o in out)
        order = sorted(range(len(fits)), key=lambda i: -fits[i])
        best_i = order[0]
        print(f"[sweep] gen {gen}: best {fits[best_i]:.3f}  mean {sum(fits)/len(fits):.3f}  "
              f"errors {errs}  over {over}  [{time.time()-t0:.0f}s]")

        with open(args.log, "a", newline="") as f:
            w = csv.writer(f)
            if new_log:
                w.writerow(["timestamp", "generation", "candidate", "fitness", "games", "seed0",
                            "config_json"])
                new_log = False
            for i, (fit, err, ov) in enumerate(out):
                w.writerow([datetime.now(timezone.utc).isoformat(), gen, i, f"{fit:.4f}",
                            args.games, args.seed0 + gen * args.games, json.dumps(cfgs[i])])

        if fits[best_i] > best_fit:
            best_fit, best_cfg = fits[best_i], cfgs[best_i]
            with open(args.best, "w") as f:
                json.dump(best_cfg, f, indent=2)
            with open(args.state, "w") as f:
                json.dump({"fitness": best_fit, "games": args.games,
                           "seed0": args.seed0, "config": best_cfg}, f, indent=2)
            print(f"[sweep] new best {best_fit:.3f} -> {args.best}")

    print(f"[sweep] done. best fitness {best_fit:.3f} in {args.best}")


if __name__ == "__main__":
    main()
