"""Fail-fast screen: play the baseline and each candidate vs F on the SAME boards/seats/dice and
report each candidate's paired difference against the baseline with a 95% CI.

    python screen.py BOARDS SEED0 PROCS base_spec cand_spec [cand_spec ...] [--clock] [--cpu_scale S]

The opponent defaults to F; set OPP=v4agent:Agent (or any module:Class) to change it.
Verdict: KEEP if diff - CI > 0, DROP if diff + CI < 0 or diff <= 0, else MAYBE (re-run bigger).
"""
import json
import os
import statistics
import sys
import time
from multiprocessing import Pool

import fastmatch


def run(pool, spec, n, seed0, flags, opp):
    jobs = [(spec, opp, seed0 + i, af, flags) for i in range(n) for af in (True, False)]
    res = pool.map(fastmatch.play_one, jobs, chunksize=2)
    return {(r["seed"], r["a_first"]): (r["winner"] == "a") for r in res}, res


def main():
    argv = sys.argv[1:]
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
    n, seed0, procs = int(rest[0]), int(rest[1]), int(rest[2])
    base, cands = rest[3], rest[4:]
    opp = os.environ.get("OPP", "f_agent:Agent")
    with Pool(procs) as pool:
        t0 = time.time()
        b, bres = run(pool, base, n, seed0, flags, opp)
        bw = sum(b.values()) / len(b)
        bs = fastmatch.summarise(bres, base)
        print(f"BASE {base}: {bw:.3f} ({len(b)} games, {time.time()-t0:.0f}s, cpu {bs['cpu']:.0f} ms/g, "
              f"net {bs['net_frac']:.2f}, over {bs['over']})", flush=True)
        for c in cands:
            t0 = time.time()
            r, rres = run(pool, c, n, seed0, flags, opp)
            d = [r[k] - b[k] for k in b]
            per_board = [d[2 * i] + d[2 * i + 1] for i in range(n)]
            mean = sum(d) / len(d)
            ci = 1.96 * statistics.pstdev(per_board) / 2 / n ** 0.5
            verdict = "KEEP" if mean - ci > 0 else ("DROP" if mean <= 0 or mean + ci < 0 else "MAYBE")
            s = fastmatch.summarise(rres, c)
            line = (f"{verdict:5s} {c}: {sum(r.values())/len(r):.3f}  diff {mean:+.3f} +-{ci:.3f}  "
                    f"cpu {s['cpu']:.0f} ms/g  net {s['net_frac']:.2f}  over {s['over']}  "
                    f"({time.time()-t0:.0f}s)")
            print(line, flush=True)
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl"), "a") as f:
                f.write(json.dumps(dict(t=time.strftime("%H:%M"), screen=c, base=base, seed0=seed0,
                                        games=len(d), win=round(sum(r.values()) / len(r), 4),
                                        base_win=round(bw, 4), diff=round(mean, 4), ci=round(ci, 4),
                                        verdict=verdict, clock=flags["clock"],
                                        cpu_ms=round(s["cpu"]),
                                        net_frac=round(s["net_frac"], 3) if s["main"] else None,
                                        over=s["over"])) + "\n")


if __name__ == "__main__":
    main()
