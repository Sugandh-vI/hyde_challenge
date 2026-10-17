# Current best Catan agent + experiment tools (17 Oct, 13:45 IST)

## sub3/: the current best submission (upload this)
Combined bot: friend 3's **agent2** plays every decision. In the main turn, friend 2's **v4 value-net ensemble** scores the same actions and overrides agent2's pick when its best action is > `MARGIN` = 0.03 higher in estimated P(win). The net is consulted only while ≥ `NET_MIN_CPU` = 0.25 s of the game's CPU allowance remains.

| File | Role |
|---|---|
| `agent.py` | entry point (the combination, ~40 lines) |
| `a2.py` | agent2: counter-opening, F-style value + Longest Road/port/dev terms, planner, robber/knight rules |
| `v4net.py`, `feats.py`, `fastcopy.py` | v4: value-net expectimax, features, fast `Game.copy` |
| `value_w0/1/2.npz` | v4 net weights (3-model ensemble) |
| `open_logreg.json` | learned opening evaluator (logistic, 20k games vs F) used by a2.py |

Zip and upload (team machine):
```
cd sub3 && python -m zipfile -c ../sub3.zip agent.py a2.py v4net.py feats.py fastcopy.py value_w0.npz value_w1.npz value_w2.npz open_logreg.json && cd .. && ~/upload.sh sub3.zip
```

## Results (pinned kit engine)
| Match | Games | Win rate | CPU/game |
|---|---|---|---|
| **sub3 vs F, official evaluate, team machine** | 800 | **79.4% ± 2.8** | 232 ms mean, 810 ms worst, 0 over clock, 0 errors |
| sub3 vs F, in-process, fresh paired seeds | 800 | 80.9% (agent2 alone 76.9%: +4.0 ± 3.1) | |
| sub3 vs agent2, official evaluate (laptop) | 200 | 50.0% ± 6.9 | 326 ms mean, 1047 ms worst |
| sub3 vs v4 | 400 | 64.2% (agent2 vs v4: 61.8%, same seeds) | |
| agent2 vs F | 400 | 75.7% | ~185-300 ms |
| v4 vs F | 400 | 70.8% | ~800 ms (loaded laptop) |

What failed (keep out): AlphaBeta depth 2 (46.5%), dev/knight-heavy weights (44%), dev bonus 300 (27.8%), naive same-turn max lookahead on F's value (46%), v4 + Longest Road nudge (−4), v4 + win finder (±0), CMA-ES over 14 weights (no gain in 5 generations). Full log: `tools/results.jsonl`, literature: `tools/research_notes.md`.

## tools/: run experiments
Set up once: Python 3.11 venv with the kit's `requirements.txt`. Run from `tools/`:
```
export KIT=<path to catan-agent-hackathon>   # the kit folder (contains catan_kit/)
export PYTHONHASHSEED=0
```
- **Paired screen (fail-fast):** candidate vs baseline on the SAME boards/seats/dice vs F, prints diff ± CI and a KEEP/DROP/MAYBE verdict:
  `python screen.py 200 560000 8 sub3agent:Agent combo:A2NetMain`
  Change the opponent with `OPP=v4agent:Agent`.
- **Head to head with game stats** (LR/LA/cities/dev buys of winners and losers):
  `python fastmatch.py sub3agent:Agent v4agent:Agent 200 560000 8`
- **Opening model:** `opengen.py` generates games vs F (`MODES=counter,score,pips`), `train_open.py open20k.npz` fits `open_logreg.json` (66% held-out accuracy from the opening alone). `open20k.npz` is the 20k-game dataset.
- `fmodel.py`: F's setup pick predictor (matches real F 93%; misses are exact ties). `check_fmodel.py` verifies it.
- Specs are `module:Class`. Use the loader modules (`sub3agent`, `v4agent`) so files named `agent.py` don't clash.

## Ideas to push further (ranked)
1. Retrain the v4 net on games played by **sub3** (policy iteration on the positions the combined bot reaches); add agent2/sub3 as training opponents.
2. Try friend 2's newer `v5` weights inside the ensemble.
3. Sweep `MARGIN` (0.01 / 0.03 / 0.06) and `NET_MIN_CPU`, using `screen.py`.
4. Check vs F **and** vs v4/agent2: robustness (worst half of the group) is part of the score.
