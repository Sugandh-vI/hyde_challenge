import os
import sys
import pathlib
# Ensure catan_kit is importable when running as script file (python my_team/run_eval.py)
# When run as "python my_team/run_eval.py" from catan-hackathon-agent/, sys.path[0] is my_team/,
# not the package root. Insert root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from catan_kit._patch import ensure_hash_seed
ensure_hash_seed()

import argparse
import csv
import time
from datetime import datetime, timezone

if __name__ == "__main__":
    import pathlib
    # Ensure we can import catan_kit when run from different cwd ( though spec says run from catan-hackathon-agent/)
    # No extra logic needed; catan_kit is in parent.

    parser = argparse.ArgumentParser(description="Wrap catan_kit.evaluate.run and log to results.csv")
    parser.add_argument("--a", required=True, help="path to agent A")
    parser.add_argument("--b", required=True, help="path to agent B")
    parser.add_argument("--games", type=int, default=100, help="number of games")
    parser.add_argument("--seed0", type=int, default=500000, help="first board seed")
    parser.add_argument("--tag", type=str, default="", help="tag for this run")
    parser.add_argument("--vps", type=int, default=10, help="vps to win")
    args = parser.parse_args()

    from catan_kit.evaluate import run

    start = time.time()
    res = run(args.a, args.b, args.games, seed0=args.seed0, vps=args.vps, quiet=False)

    # Also print wall time already printed by run()
    # Append to results.csv
    # fields: timestamp, tag, agent A, agent B, games, seed0, win_rate, ci95, mean_cpu_ms, worst_cpu_ms, games_over_budget, errors, timeouts, crashes
    timestamp = datetime.now(timezone.utc).isoformat()
    # Also could use local time; use utc isoformat
    row = {
        "timestamp": timestamp,
        "tag": args.tag,
        "agent_a": args.a,
        "agent_b": args.b,
        "games": args.games,
        "seed0": args.seed0,
        "win_rate": res.get("win_rate", ""),
        "ci95": res.get("ci95", ""),
        "mean_cpu_ms": round(res.get("mean_cpu_s", 0) * 1000, 2) if "mean_cpu_s" in res else "",
        "worst_cpu_ms": round(res.get("worst_cpu_s", 0) * 1000, 2) if "worst_cpu_s" in res else "",
        "games_over_budget": res.get("games_over_budget", ""),
        "errors": res.get("errors", ""),
        "timeouts": res.get("timeouts", ""),
        "crashes": res.get("crashes", ""),
    }

    csv_path = os.path.join(os.path.dirname(__file__), "results.csv")
    header = ["timestamp","tag","agent_a","agent_b","games","seed0","win_rate","ci95","mean_cpu_ms","worst_cpu_ms","games_over_budget","errors","timeouts","crashes"]
    file_exists = os.path.exists(csv_path)
    # If file doesn't exist, write header; else append
    try:
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists or os.path.getsize(csv_path) == 0:
                # check if header needed: if file newly created or empty
                # Also need to detect if header already exists by peeking first line
                # Simple: if not file_exists or size==0, write header
                writer.writeheader()
            else:
                # verify header present; if file exists but header missing, we still write row without header
                # To be safe, we peek first line
                try:
                    with open(csv_path, "r") as rf:
                        first = rf.readline().strip()
                        if not first or first.split(",")[0] != "timestamp":
                            # header missing, but we already opened for append, need to handle
                            pass
                except Exception:
                    pass
            writer.writerow(row)
        print(f"[run_eval] appended to {csv_path}")
    except Exception as e:
        print(f"[run_eval] failed to write csv: {e}", file=sys.stderr)

    # Also print a parsable line for scripts
    print(f"[run_eval] tag={args.tag} win_rate={row['win_rate']:.4f} mean_cpu_ms={row['mean_cpu_ms']} worst_cpu_ms={row['worst_cpu_ms']} over_budget={row['games_over_budget']}")
