"""Fail-fast screen: play the baseline and each candidate vs F on the SAME boards/seats/dice, report
each candidate's paired difference vs baseline with a 95% CI. One line per candidate, logged.
Usage: python screen.py BOARDS SEED0 PROCS base_spec cand_spec [cand_spec ...]
Verdict: KEEP if diff - CI > 0, DROP if diff + CI < 0 or diff <= 0, else MAYBE (re-run bigger).
"""
import json
import os
import statistics
import sys
import time
from multiprocessing import Pool

import fastmatch


def run(pool, spec, n, seed0):
    jobs = [(spec, os.environ.get("OPP", "f_agent:Agent"), seed0 + i, af) for i in range(n) for af in (True, False)]
    res = pool.map(fastmatch.play_one, jobs, chunksize=2)
    return {(r["seed"], r["a_first"]): (r["winner"] == "a") for r in res}, res


def main():
    n, seed0, procs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    base, cands = sys.argv[4], sys.argv[5:]
    with Pool(procs) as pool:
        t0 = time.time()
        b, bres = run(pool, base, n, seed0)
        bw = sum(b.values()) / len(b)
        cpu = sum(r["a_cpu"] for r in bres) / len(bres)
        print(f"BASE {base}: {bw:.3f} ({len(b)} games, {time.time()-t0:.0f}s, cpu {1000*cpu:.0f} ms/g)", flush=True)
        for c in cands:
            t0 = time.time()
            r, rres = run(pool, c, n, seed0)
            d = [r[k] - b[k] for k in b]
            per_board = [d[2 * i] + d[2 * i + 1] for i in range(n)]
            mean = sum(d) / len(d)
            ci = 1.96 * statistics.pstdev(per_board) / 2 / n ** 0.5
            verdict = "KEEP" if mean - ci > 0 else ("DROP" if mean <= 0 or mean + ci < 0 else "MAYBE")
            cpu = sum(x["a_cpu"] for x in rres) / len(rres)
            line = (f"{verdict:5s} {c}: {sum(r.values())/len(r):.3f}  diff {mean:+.3f} +-{ci:.3f}  "
                    f"cpu {1000*cpu:.0f} ms/g  ({time.time()-t0:.0f}s)")
            print(line, flush=True)
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl"), "a") as f:
                f.write(json.dumps(dict(t=time.strftime("%H:%M"), screen=c, base=base, seed0=seed0,
                                        games=len(d), win=round(sum(r.values()) / len(r), 4),
                                        base_win=round(bw, 4), diff=round(mean, 4), ci=round(ci, 4),
                                        verdict=verdict)) + "\n")


if __name__ == "__main__":
    main()
