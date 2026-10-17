hellooo
---

## Arena update 1 — baseline, tools, same-turn search, weight tuner

### 1. What I built / changed

All paths are relative to `catan-hackathon-agent/`.

| File | One line |
|---|---|
| `my_team/agent_v1.py` | Safe copy of F: one-step `base_fn` evaluation, immediate-win check (actual VPs ≥ 10), try/except around everything, index 0 fallback. |
| `my_team/agent_v2.py` | Agent v2: F's value function + same-turn iterative-deepening beam search + CPU budget. Standalone entry point. |
| `my_team/search.py` | v2's helper: custom candidate pruning, hidden-VP-aware value function, chance-node expansion (`execute_spectrum`, corrected robber probabilities), budget, search. |
| `my_team/agent.py` | Current submission: an exact copy of `agent_v2.py`. |
| `my_team/weights.json` | `catanatron`'s `DEFAULT_WEIGHTS` in the tunable JSON format (this is F's weight vector). |
| `my_team/run_eval.py` | Wraps `catan_kit.evaluate.run`, prints the summary, appends one row to `my_team/results.csv`. |
| `my_team/make_submission.sh` | Builds `submission/` and `my_team.zip` with `agent.py`, helper `.py`, `weights.json` at the zip's top level. |
| `my_team/tune_weights.py` | (mu, lambda) evolution strategy in numpy that tunes the 13 weights on win rate vs F; parallel, resumable, `--smoke`. |
| `my_team/EXPERIMENTS.md` | Experiment log: hypothesis, setup, seeds, result, conclusion for every run below. |
| `my_team/results.csv` | Machine-readable log of every harness run (not shipped in the zip). |
| `catan-hackathon-agent/.gitignore` | Ignores `submission/` and `my_team.zip`. |

Nothing in `catan_kit/`, `examples/` or `requirements.txt` was touched.

### 2. Design decisions and why

* **Depth 1 is always completed.** The first version checked the clock inside the one-step
  evaluation loop and could return the best of a truncated prefix of the action list. With K=2
  worlds this scored **0.125** against F — a catastrophic, silent failure. The depth-1 pass now
  never stops early (it costs about what F costs, and the allowance is ~6× F), and only the
  optional deepening is deadline-limited. This was the single most important bug found.
* **Pruning keeps every robber move.** `catanatron`'s `list_prunned_actions` keeps only one robber
  tile chosen by a production heuristic. Measured: it changed only `MOVE_ROBBER` choices, with a
  relative value loss of ~3e-8, but cost ~10 points of win rate. Replaced by a custom pass that
  removes only redundant maritime trades (duplicates, 4:1 when a 3:1 port exists).
* **Hidden victory-point cards are valued.** `base_fn` scores public VPs only, so a VP development
  card in our hand is invisible to F. We know our own dev cards exactly (`obs.my_devs`), so
  `(ACTUAL − PUBLIC) × public_vps` is added to the value. This also makes the win-check consistent.
* **Chance nodes use `execute_spectrum`**, probability-weighted. For robber steals the engine uses
  a uniform 1/5; in 2-player Catan the victim's hand is exactly derivable from the bank and our
  hand, so the real probabilities (card counts / hand size) are used instead.
* **Budget:** `budget = (obs.cpu_left − 250 ms) / 30`, capped at 25 ms (60 ms in the opening),
  multiplied by 0.8 as an internal safety margin; the opening (initial placement) gets a larger cap
  because it has the most legal moves. The 250 ms reserve means the clock is never exhausted.
* **Clock safety net:** if `obs.cpu_left` is ever below 3 ms, `decide()` returns `END_TURN` when it
  is legal (else index 0) instead of starting a ~10 ms evaluation that could tip the harness into
  playing random moves for the rest of the game.
* **Search:** depth 1 = F; deeper levels deepen only the 6 best root actions, keep the best 3
  children per node, stop at `END_TURN` / a change of turn, and average chance nodes. Iterative
  deepening returns the best move of the deepest *completed* level.
* **K = 1 world by default** (as specified). K is a knob and K=2 was measured (E8); it is not
  better at 40 games but costs ~2× CPU.
* **Weights stay at `DEFAULT_WEIGHTS`.** A hand-calibrated rescaling (1 VP = 1.0, production
  0.15 VP/unit, cards 0.06 VP) scored 0.30–0.33 against F and was rejected. Tuning must be
  empirical; that is what `tune_weights.py` is for.
* **Errors:** every path is wrapped; `decide()` never raises and always returns an in-range index.
  Across all runs below: **0 errors, 0 timeouts, 0 crashes, 0 games over budget**.

### 3. Exact commands

Run everything from `catan-hackathon-agent/` (Python 3.11 venv with `requirements.txt`).

```bash
cd catan-hackathon-agent

# install check
python -m catan_kit.selfcheck

# baseline (F copy) and the current agent vs F
python my_team/run_eval.py --a my_team/agent_v1.py --b examples/f_agent.py --games 20  --seed0 500000 --tag v1_vs_F
python my_team/run_eval.py --a my_team/agent.py    --b examples/f_agent.py --games 40  --seed0 500000 --tag v2_vs_F
python my_team/run_eval.py --a my_team/agent.py    --b examples/random_agent.py --games 10 --seed0 500000 --tag v2_vs_random

# real tuning (NOT run here): 200 games/candidate, pop 16, 20 generations, 4 workers
python my_team/tune_weights.py --games 200 --pop 16 --generations 20 --workers 4 --sigma 0.4
#   resumable:  add --resume my_team/weights_tuned.json
#   start from a hand-written vector: add --init my_team/weights_cal.json
#   the best-so-far is written to my_team/weights_tuned.json after every generation

# tuner smoke test (this one was run)
python my_team/tune_weights.py --smoke --workers 2

# build the submission zip (agent.py at the top level)
bash my_team/make_submission.sh
```

After tuning, copy the tuned vector into `my_team/weights.json` (or set the agent to read
`weights_tuned.json`) and re-run the evaluation. `make_submission.sh` ships whichever
`weights.json` is present.

### 4. Test results actually obtained

Seeds are `seed0 = 500000` (dev convention). Full detail in `my_team/EXPERIMENTS.md`; every harness
run is also in `my_team/results.csv`.

`python -m catan_kit.selfcheck` — all checks passed; F measured at 350 ms mean CPU/game here.

```
A = my_team/agent_v1.py       B = examples/f_agent.py     (20 games)
A win rate          0.5000 +-0.2191   (10/20)
A mean CPU / game      385.5 ms
A worst CPU / game     880.0 ms
A games over budget        0
A errors                   0      (of 1271 decisions)
A timeouts/crashes    0 / 0
```

```
A = my_team/agent.py          B = examples/f_agent.py     (40 games, shipped config K=1 depth=3)
A win rate          0.5250 +-0.1548   (21/40)
A mean CPU / game      481.8 ms
A worst CPU / game     970.0 ms
A games over budget        0
A errors                   0      (of 2114 decisions)
A timeouts/crashes    0 / 0
```

```
A = my_team/agent.py          B = examples/random_agent.py (10 games)
A win rate          1.0000 +-0.0000   (10/10)
A mean CPU / game      560.0 ms
A games over budget        0
A errors                   0
```

```
A = /tmp/subtest/agent.py     B = examples/f_agent.py      (10 games, unzipped submission)
A win rate          0.5000 +-0.3099   (5/10)
A mean CPU / game      525.0 ms
A games over budget        0
A errors                   0
A timeouts/crashes    0 / 0
```

Final sanity after the clock-guard change and the rebuilt zip:

```
A = my_team/agent.py          B = examples/f_agent.py      (10 games)
A win rate          0.6000 +-0.3036   (6/10)
A mean CPU / game      509.0 ms
A worst CPU / game     750.0 ms
A games over budget        0
A errors                   0      (of 577 decisions)
A timeouts/crashes    0 / 0

A = /tmp/subtest2/agent.py    B = examples/random_agent.py (6 games, unzipped submission)
A win rate          1.0000 +-0.0000   (6/6)
A games over budget        0
A errors                   0
```

`tune_weights.py --smoke --workers 2`: 2 generations × 4 candidates × 10 games, ~32 s; fitness
0.40 → 0.50 (10-game noise); `tuning_log.csv` and `weights_tuned.json` written correctly.

Depth / K ablation under the harness (40 games each): depth 1 = 0.450, depth 2 = 0.375,
depth 3 = 0.400 *before* the truncated-pass fix; after the fix depth 1 = 0.450, K=2 depth 1 = 0.500,
depth 3 = 0.525. **All 40-game CIs are ±0.15, so none of these differences is statistically
significant.** The honest summary is: v2 is at F's level, not yet clearly above it.

### 5. Known issues, risks and open questions

* **CPU risk is low but the timing extrapolation is an estimate.** On this sandbox F itself costs
  350 ms/game (the scoring machine: 158 ms; the reference laptop: ~97 ms). Our agent costs
  482 ms/game here, i.e. **~1.4× F**, which extrapolates to roughly 220 ms on the scoring machine
  and ~135 ms on a laptop — well under the 450 ms target and the 1.5 s cap. Worst observed game was
  970 ms here (~440 ms on the scoring machine). Still: check the leaderboard's "games over the
  clock" column after the first upload.
* **Strength is not established.** 40 games cannot resolve a 5-point difference. v2's search has
  not yet beaten F convincingly; the main lever left is weight tuning.
* **The tuner's log-space step is slow.** The weights span ~14 orders of magnitude; `sigma = 0.4`
  changes a weight ~1.5× per generation. Use `--init` for a structurally different scale.
* **K > 1 averages only deterministic actions across worlds**; chance actions (ROLL, BUY, robber)
  use world 0 for speed. Worth fixing if K=2 becomes the default.
* **The search's root deepening skips chance actions** (rolling before/after a build reaches the
  same position within a turn), which is what keeps the CPU low.
* **Open questions for the team:** (a) should the shipped default be depth 3 (0.525, 482 ms) or
  K=2 depth 1 (0.500, 574 ms)? (b) how many games per candidate can your machine afford for the
  real tuning run? (c) do you want `agent.py` to read `weights_tuned.json` directly once it exists?

### 6. Suggested next experiments

1. **Run the real tuner** (200 games/candidate, pop 16, 20+ generations, workers 4–6 on the 8-core
   machine), starting from `DEFAULT_WEIGHTS`. Evaluate the winner on 1,000 fresh seeds
   (`seed0 = 990000`) before shipping.
2. **Scale sweep on one weight at a time** (cheap, 40–100 games each): `hand_resources`,
   `hand_devs`, `army_size`, `production`, each multiplied by 10/100/1000 relative to the default.
   The hand-calibrated vector failed, but a single-factor sweep may find the direction.
3. **K=2 at depth 3** — CPU would be ~700 ms here (~320 ms on the scoring machine), still inside
   budget; it combines the two effects that were individually neutral.
4. **Depth 2 only in the opening** (the first four placements) where search is cheap and the
   decisions matter most.
5. **Opponent-aware blocking:** add the opponent's public VPs (and the number of unplayed dev
   cards) to the value function. Within-turn search does not see it, but the robber and settlement
   choices might improve.
6. **Port/robber features** not in the 13: owning a 2:1 port, robber on our best tile.
