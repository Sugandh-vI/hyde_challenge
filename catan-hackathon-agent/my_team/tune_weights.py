"""Tune the 13 value-function weights on win rate against F.

Why this file exists
--------------------
F's weights (``catanatron.players.value.DEFAULT_WEIGHTS``) are not tuned for 2-player Catan.  The
one-step agent is cheap (about F's own CPU cost), so it is the right search-free fitness function
for tuning: every candidate gets a temp folder with its own ``weights.json`` and plays N games
against F on dev seeds, with seats alternated by the evaluation harness.

Parameterisation
----------------
Weights move in *relative* (log) space: ``w = base * exp(delta)``.  Positive and negative weights
keep their sign; a zero base (``reachable_production_0``) uses a linear map instead.  The search is
a simple (mu, lambda) evolution strategy in numpy: sample lambda candidates around the mean,
evaluate, keep the top mu, set the new mean to their average, adapt sigma with a 1/5-style rule.
``--sigma`` defaults to 0.40 (a factor of ~1.5 per generation).  A caution from the experiments:
the 13 weights span many orders of magnitude (``public_vps`` is 3e14, ``num_tiles`` is 1), so a
log-space step of 0.4 still needs ~30 generations to change a weight by 1e5.  If you want to test a
structurally different scale, start from a hand-written ``--init`` file and tune from there.

Usage
-----
    python my_team/tune_weights.py --smoke                    # 2 generations, pop 4, 10 games
    python my_team/tune_weights.py --games 200 --pop 16 --generations 20 --workers 4
    python my_team/tune_weights.py --init my_team/weights_cal.json --sigma 0.3

The best weights so far are written to ``my_team/weights_tuned.json`` after every generation, so
the run can be stopped at any time; ``my_team/tuning_log.csv`` gets one row per generation.
"""

import os
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from catan_kit._patch import ensure_hash_seed
ensure_hash_seed()

import json
import csv
import argparse
import tempfile
import shutil
import random
import datetime
from datetime import timezone

# Must be under __main__ guard for multiprocessing on some platforms

# Global defaults for weight keys order
WEIGHT_KEYS = [
    "public_vps",
    "production",
    "enemy_production",
    "num_tiles",
    "reachable_production_0",
    "reachable_production_1",
    "buildable_nodes",
    "longest_road",
    "hand_synergy",
    "hand_resources",
    "discard_penalty",
    "hand_devs",
    "army_size",
]

def load_default_weights():
    try:
        from catanatron.players.value import DEFAULT_WEIGHTS
        return DEFAULT_WEIGHTS
    except Exception:
        return {
            "public_vps": 300000000000000.0,
            "production": 100000000.0,
            "enemy_production": -100000000.0,
            "num_tiles": 1,
            "reachable_production_0": 0,
            "reachable_production_1": 10000.0,
            "buildable_nodes": 1000.0,
            "longest_road": 10,
            "hand_synergy": 100.0,
            "hand_resources": 1,
            "discard_penalty": -5,
            "hand_devs": 10,
            "army_size": 10.1,
        }

def weights_to_vector(weights):
    return [weights[k] for k in WEIGHT_KEYS]

def vector_to_weights(vec):
    return {k: float(v) for k, v in zip(WEIGHT_KEYS, vec)}

def deltas_to_weights(deltas, base_weights, sigma_scale=1.0):
    """
    Convert log-space deltas (vector length 13) to weights dict.
    We optimize in log-space multiplicative: w = base * exp(delta)
    For base == 0, we use additive: w = delta * scale (scale=1000)
    This keeps signs: negative bases stay negative (exp >0 preserves sign).
    """
    new = {}
    for i, k in enumerate(WEIGHT_KEYS):
        base = base_weights[k]
        d = deltas[i]
        if abs(base) < 1e-9:
            # Special handling for zero weight (reachable_production_0)
            # Use scale 1000, allow both positive and negative via delta
            # We map delta -> weight as delta*500 + small exp term to allow smooth
            # Use: w = 500 * d  (if we want zero mean centered)
            # But to keep log-space spirit, we interpret d as additive in linear space scaled
            # Let's use: w = 1000 * (exp(d) - 1)  -> 0 when d=0, ~1000 when d>0, -1000 when d<0 (but exp(d)-1 >= -1, so min -1000)
            # That's limited to -1000 lower bound; instead use linear: w = 1000*d
            scale = 1000.0
            # For this key we want to explore around 0 with sigma scaling
            # Use linear mapping: w = d * scale * 2  (since d ~ N(0, sigma^2), sigma~0.2 => w range ~ +-400)
            new[k] = float(d * scale)
        else:
            # multiplicative
            # delta is drawn from N(0, sigma^2) where sigma ~0.2-0.5 gives factor exp(delta) in [0.6, 1.6]
            try:
                factor = math.exp(d)
            except OverflowError:
                factor = math.exp(5) if d > 0 else math.exp(-5)
            new[k] = float(base * factor)
    return new

# Need math import for exp
import math

# Worker function for parallel evaluation
def evaluate_candidate_worker(args_tuple):
    """
    Worker for multiprocessing Pool.
    args_tuple = (candidate_weights_dict, source_agent_path, opponent_path, games, seed0, tmp_base)
    Returns win_rate
    """
    # Set single thread for torch/numpy to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    candidate_weights, source_agent_path, opponent_path, games, seed0 = args_tuple

    # Create temp folder with copy of agent files + candidate weights.json
    tmpdir = tempfile.mkdtemp(prefix="catan-tune-")
    try:
        # Determine source team dir
        source_dir = os.path.dirname(os.path.abspath(source_agent_path))
        # Copy source agent as agent.py
        dest_agent = os.path.join(tmpdir, "agent.py")
        shutil.copyfile(source_agent_path, dest_agent)

        # Copy helper .py files from source_dir (search.py etc.) if they exist, excluding scripts
        # We copy any .py that might be needed: search.py, any other helper
        exclude = {"run_eval.py", "tune_weights.py", "agent_v1.py", "agent_v2.py"}
        try:
            for fname in os.listdir(source_dir):
                if fname.endswith(".py") and fname not in exclude:
                    src = os.path.join(source_dir, fname)
                    dst = os.path.join(tmpdir, fname)
                    # Don't overwrite the dest_agent if it's same as one of these (e.g., source is agent.py)
                    if os.path.abspath(src) == os.path.abspath(source_agent_path):
                        continue
                    # Only copy if not already exists and is file
                    if os.path.isfile(src) and not os.path.exists(dst):
                        shutil.copyfile(src, dst)
        except Exception:
            pass

        # Write candidate weights.json
        with open(os.path.join(tmpdir, "weights.json"), "w") as f:
            json.dump(candidate_weights, f, indent=2)

        # Also copy existing weights.json if candidate wants to reuse? Already wrote.

        # Evaluate vs F
        # Need to import here to avoid issues with spawn
        from catan_kit.evaluate import run as eval_run
        # Use quiet True to suppress output per worker
        result = eval_run(dest_agent, opponent_path, games, seed0=seed0, quiet=True)
        # result is dict with win_rate for side a (candidate)
        win_rate = result.get("win_rate", 0.0)
        # Also consider that if candidate crashes a lot, win_rate will be low but we also penalize
        # We could incorporate errors, but simple win_rate is fitness
        return float(win_rate)
    except Exception as e:
        # On any error, return 0
        # print(f"worker error: {e}", file=sys.stderr)
        return 0.0
    finally:
        # Cleanup temp dir
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(description="Tune value-function weights via ES vs F")
    parser.add_argument("--agent", type=str, default="my_team/agent_v1.py", help="path to agent to tune (cheap one-step agent)")
    parser.add_argument("--games", type=int, default=200, help="games per fitness evaluation")
    parser.add_argument("--pop", type=int, default=16, help="population size (lambda)")
    parser.add_argument("--generations", type=int, default=15, help="number of generations")
    parser.add_argument("--workers", type=int, default=7, help="parallel workers (multiprocessing Pool)")
    parser.add_argument("--seed0", type=int, default=500000, help="first board seed for evaluation")
    parser.add_argument("--sigma", type=float, default=0.40, help="initial sigma (step size in log-space)")
    parser.add_argument("--smoke", action="store_true", help="smoke test: 2 generations, pop 4, 10 games")
    parser.add_argument("--opponent", type=str, default="examples/f_agent.py", help="opponent agent path (usually F)")
    parser.add_argument("--resume", type=str, default=None, help="optional path to weights_tuned.json to resume from")
    parser.add_argument("--init", type=str, default=None, help="optional weights JSON to center the search on (default: DEFAULT_WEIGHTS)")
    parser.add_argument("--log", type=str, default="my_team/tuning_log.csv", help="log csv path")
    parser.add_argument("--best", type=str, default="my_team/weights_tuned.json", help="best weights json path")
    args = parser.parse_args()

    # Enforce single thread in main as well (though main will spawn workers)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    # Smoke override
    if args.smoke:
        print("[tune] SMOKE mode: overriding to 2 generations, pop 4, 10 games")
        args.generations = 2
        args.pop = 4
        args.games = 10
        args.workers = min(args.workers, 4)

    # Resolve paths relative to catan-hackathon-agent root (where this script is expected to be run from)
    # But we also handle when cwd is different: make paths absolute based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)  # catan-hackathon-agent/
    # Resolve agent path
    def resolve_path(p):
        if os.path.isabs(p):
            return p
        # Try relative to cwd
        if os.path.exists(p):
            return os.path.abspath(p)
        # Try relative to root_dir
        cand = os.path.join(root_dir, p)
        if os.path.exists(cand):
            return cand
        # Try relative to script_dir
        cand2 = os.path.join(script_dir, p)
        if os.path.exists(cand2):
            return cand2
        return os.path.abspath(p)

    agent_path = resolve_path(args.agent)
    opponent_path = resolve_path(args.opponent)
    log_path = resolve_path(args.log)
    best_path = resolve_path(args.best)

    print(f"[tune] agent={agent_path}")
    print(f"[tune] opponent={opponent_path}")
    print(f"[tune] games={args.games} pop={args.pop} generations={args.generations} workers={args.workers} seed0={args.seed0} sigma={args.sigma}")
    if not os.path.exists(agent_path):
        print(f"[tune] ERROR: agent path not found: {agent_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(opponent_path):
        print(f"[tune] ERROR: opponent path not found: {opponent_path}", file=sys.stderr)
        sys.exit(1)

    base_weights = load_default_weights()
    if args.init:
        try:
            init_path = resolve_path(args.init)
            with open(init_path, "r") as f:
                init_weights = json.load(f)
            for k in WEIGHT_KEYS:
                if k in init_weights:
                    base_weights[k] = float(init_weights[k])
            print(f"[tune] centered on --init weights from {init_path}")
        except Exception as e:
            print(f"[tune] could not read --init ({e}); using DEFAULT_WEIGHTS", file=sys.stderr)
    # If resume weights provided, load them as new base? Or use them as mean start
    initial_mean = [0.0] * len(WEIGHT_KEYS)
    best_overall_weights = base_weights
    best_overall_fitness = 0.0

    if args.resume and os.path.exists(args.resume):
        try:
            with open(args.resume, "r") as f:
                resumed = json.load(f)
                # If resumed is dict of weights, we need to convert to deltas relative to base
                # Compute deltas that would produce resumed weights
                deltas = []
                for k in WEIGHT_KEYS:
                    base = base_weights[k]
                    target = resumed.get(k, base)
                    if abs(base) < 1e-9:
                        # for zero weight, delta = target / 1000
                        d = target / 1000.0
                    else:
                        # multiplicative: factor = target/base, delta = log(factor)
                        try:
                            factor = target / base
                            if factor <= 0:
                                factor = 1e-6
                            d = math.log(factor)
                        except Exception:
                            d = 0.0
                    deltas.append(d)
                initial_mean = deltas
                best_overall_weights = resumed
                print(f"[tune] resumed from {args.resume} -> initial mean set")
        except Exception as e:
            print(f"[tune] resume failed: {e}, starting from zero mean")

    # ES parameters
    pop = args.pop
    generations = args.generations
    sigma = args.sigma
    # mu = floor(pop/2) or pop//4, use pop//3
    mu = max(1, pop // 3)
    print(f"[tune] mu={mu} lambda={pop} initial sigma={sigma}")

    # Prepare log file header
    log_exists = os.path.exists(log_path)
    if not log_exists:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "best_fitness", "mean_fitness", "sigma", "weights_json"])
    else:
        # ensure header exists if file empty
        if os.path.getsize(log_path) == 0:
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["generation", "best_fitness", "mean_fitness", "sigma", "weights_json"])

    # History for best save
    import numpy as np

    # For reproducibility, seed numpy
    np.random.seed(12345)
    random.seed(12345)

    mean = np.array(initial_mean, dtype=float)

    # Evaluate initial baseline (base weights) as reference?
    # We could but not necessary; just start evolution.

    # Main loop
    import multiprocessing as mp
    # Ensure we use spawn? But we already called ensure_hash_seed, and we are in main guard.
    # Use mp.Pool
    # Note: on some systems fork after importing catan_kit may be okay, but we prefer spawn for safety?
    # However spawn re-imports everything and may re-trigger ensure_hash_seed restart.
    # We'll use default context (fork on Linux).

    for gen in range(generations):
        gen_start = datetime.datetime.now(timezone.utc).isoformat()
        print(f"\n[tune] Generation {gen} / {generations}  mean norm={np.linalg.norm(mean):.3f} sigma={sigma:.3f}  ({gen_start})")

        # Sample pop candidates: deltas = mean + sigma * N(0,1)
        candidates_deltas = []
        candidates_weights = []
        for i in range(pop):
            delta = mean + sigma * np.random.randn(len(WEIGHT_KEYS))
            candidates_deltas.append(delta)
            w = deltas_to_weights(delta, base_weights)
            candidates_weights.append(w)

        # Prepare worker args
        worker_args = []
        for w in candidates_weights:
            # For diversity, each candidate could use slightly different seed0 offset?
            # Use seed0 + gen*1000 + idx offset? But spec says dev seeds seed0=500000+, so we keep same seed0 for fairness within generation.
            # However to avoid identical dice across generations, we can offset by gen*10000.
            # Let's use seed0 + gen*1000 (so each generation uses different board set, average over many boards).
            seed_for_eval = args.seed0 + gen * 1000  # keep same within generation for fair comparison
            worker_args.append((w, agent_path, opponent_path, args.games, seed_for_eval))

        # Evaluate in parallel
        # Use Pool with workers
        # For smoke, we want quick; use pool anyway
        try:
            # ProcessPoolExecutor, NOT mp.Pool: Pool workers are daemonic and Python refuses to let
            # a daemonic process create children, which catan_kit.match must do (one OS process per
            # agent).  ProcessPoolExecutor workers are ordinary processes and can spawn.
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
                fitnesses = list(pool.map(evaluate_candidate_worker, worker_args))
        except Exception as e:
            print(f"[tune] worker pool failed ({e}), falling back to serial", file=sys.stderr)
            fitnesses = []
            for arg in worker_args:
                fitnesses.append(evaluate_candidate_worker(arg))

        fitnesses = np.array(fitnesses, dtype=float)
        # Handle NaNs
        fitnesses = np.nan_to_num(fitnesses, nan=0.0)

        # Rank: descending fitness (win rate)
        sorted_idx = np.argsort(-fitnesses)
        best_idx = sorted_idx[0]
        best_fitness = float(fitnesses[best_idx])
        mean_fitness = float(np.mean(fitnesses))
        best_weights = candidates_weights[best_idx]
        best_delta = candidates_deltas[best_idx]

        print(f"[tune] gen {gen} best={best_fitness:.4f} mean={mean_fitness:.4f} worst={float(np.min(fitnesses)):.4f}")
        # Show best delta
        try:
            print(f"[tune] best weights: {json.dumps(best_weights, indent=2)}")
        except Exception:
            pass

        # Update best overall if improved
        if best_fitness > best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_weights = best_weights
            print(f"[tune] New overall best {best_overall_fitness:.4f} at gen {gen}")

        # Selection: top mu to update mean
        top_mu_idx = sorted_idx[:mu]
        # New mean = average of top mu deltas
        top_deltas = [candidates_deltas[i] for i in top_mu_idx]
        new_mean = np.mean(top_deltas, axis=0)

        # Simple sigma adaptation: if best improved over previous mean fitness, keep sigma, else decay slightly
        # We track previous best; for simplicity, we decay sigma slowly to converge
        # Or 1/5 success rule: if best > median, increase sigma else decrease
        median_fitness = float(np.median(fitnesses))
        # Count how many of top mu are above median? Actually all top are above median if mu < pop/2
        # Use simple rule: if best_fitness > mean_fitness + 0.02, increase sigma a bit, else decrease
        if best_fitness > mean_fitness + 0.02:
            sigma *= 1.02
        else:
            sigma *= 0.98
        # Clamp sigma
        sigma = max(0.05, min(1.0, sigma))

        mean = new_mean

        # Log generation
        try:
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                # weights as json string (compact)
                weights_json_str = json.dumps(best_weights)
                writer.writerow([gen, f"{best_fitness:.5f}", f"{mean_fitness:.5f}", f"{sigma:.5f}", weights_json_str])
        except Exception as e:
            print(f"[tune] Failed to log: {e}", file=sys.stderr)

        # Save best overall after every generation (so it can be stopped anytime)
        try:
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            with open(best_path, "w") as out:
                json.dump(best_overall_weights, out, indent=2)
            print(f"[tune] Saved best overall weights to {best_path} (fitness {best_overall_fitness:.4f})")
        except Exception as e:
            print(f"[tune] Failed to save best: {e}", file=sys.stderr)

        # Also save current gen best as separate? No need

        # Early stopping if sigma very small?
        # Continue

    print(f"\n[tune] Done. Best overall fitness {best_overall_fitness:.4f}")
    print(f"[tune] Best weights saved to {best_path}")
    try:
        print(json.dumps(best_overall_weights, indent=2))
    except Exception:
        pass

