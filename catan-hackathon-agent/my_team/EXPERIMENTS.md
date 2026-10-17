# Experiments log

One entry per experiment: hypothesis, setup, seeds, result, conclusion.

Harness notation:
* **harness** = `catan_kit.evaluate.run` / `my_team/run_eval.py`, both agents in separate OS
  processes, CPU clock enforced (this is the scorer).
* **in-process** = `/tmp/arena.py` (scratch driver, not shipped): both agents and the game share
  one process, no clock, alternating seats and paired dice per board. Used only to triage ideas
  quickly. It is *indicative*: chance actions executed during the search draw from the global
  `random`, which is shared with the dice in this mode (in the harness the dice live in the
  harness process), so different variants see slightly different dice streams.
* Dev seeds are `seed0 = 500000` throughout; 40 games = 20 boards, each played from both seats.
* `F` = `examples/f_agent.py`. `v1` = `my_team/agent_v1.py` (safe copy of F). `v2` =
  `my_team/agent_v2.py` + `my_team/search.py` (same-turn search).

All CPU numbers are from this development sandbox. **The sandbox is slow**: the kit's own
selfcheck measures F at ~350 ms/game here, versus 158 ms/game on the scoring machine and ~97 ms on
the reference laptop. Normalise by F's number before comparing with the 450 ms target.

---

## E0. Install self-check

* **Setup:** `python -m catan_kit.selfcheck` (Python 3.11.2, venv, requirements.txt + CPU torch).
* **Result:** all checks passed. Engine determinism reproduced (`seed 4256 -> ('Color.RED', 111)`).
  F beats random 20/20, 350 ms mean CPU/game here.
* **Conclusion:** environment is trustworthy; local numbers are directly comparable to the kit's.

## E1. v1 = safe F copy

* **Hypothesis:** a copy of F with a win-check and an exception fallback scores 0.5 against F
  (identical policies, seats alternate).
* **Setup:** harness, `v1 vs examples/f_agent.py`, 20 games, seed0 500000.
* **Result:** **0.500 ± 0.219**, 385.5 ms mean CPU/game, 880 ms worst, 0 over budget, 0 errors,
  0 timeouts.
* **Conclusion:** the safety wrapper costs about 10 % CPU over F (385 vs 350 ms) and changes
  nothing statistically. Baseline established.

## E2. v2 first cut (all features at once)

* **Hypothesis:** same-turn search (depth 3, beam 3), catanatron's `list_prunned_actions`, and a
  hidden-VP-aware value function beat F.
* **Setup:** harness, `v2 vs F`, 40 games, seed0 500000. Also `v2 vs random`, 10 games.
* **Result:** 0.400 ± 0.152 vs F; 1.000 vs random (0 errors). 612.8 ms CPU/game, 1120 ms worst,
  0 over budget.
* **Conclusion:** no improvement, and CPU 1.75× F. The result is not conclusive alone, but it
  motivated measuring each component separately (E4–E7).

## E3. In-process component isolation (v1 + one change at a time)

* **Hypothesis:** the regression comes from a single component.
* **Setup:** in-process arena, 40 games per variant, same seeds; variants: `v_hidden` = v1 + hidden
  victory-point cards in the value function; `v_prune` = v1 but only over `list_prunned_actions`
  candidates; `v1 vs v1` sanity.
* **Result:**
  * v1 vs v1: 12/24 = **0.500** (driver sanity).
  * `v_hidden` vs v1: **0.525** (21/40).
  * `v_prune` vs v1: **0.375** (15/40).
  * Diagnostic: pruning changed the chosen action in 55 of ~8 games' decisions and **every single
    change was a `MOVE_ROBBER` tile**; the relative value loss was ~3e-8, i.e. invisible to the
    value function but path-changing.
* **Conclusion:** `list_prunned_actions`' robber pruning (keeps only the "most impactful" tile) is
  a bad idea for this game: it systematically changes robber play for a negligible CPU saving.
  Hidden-VP scoring is neutral-to-positive. → replaced pruning with a custom pass that keeps all
  robber moves and only removes redundant maritime trades.

## E4. Depth ablation, in-process

* **Hypothesis:** deeper same-turn search helps.
* **Setup:** in-process, no pruning, base value function (no hidden VP), MAX_DEPTH ∈ {1,2,3},
  40–60 games per variant vs v1.
* **Result:** depth 1 **0.500** (20/40), depth 2 **0.450** (18/40), depth 3 **0.517** (31/60);
  depth 3 with maritime trades removed **0.467** (28/60).
* **Conclusion:** the search is roughly neutral in-process; removing trades from the candidate set
  does not help, so "the search wastes cards on trades" was not the explanation.

## E5. Depth ablation, harness (ground truth, before the fix in E8)

* **Hypothesis:** same as E4, measured under the real clock and process isolation.
* **Setup:** harness, 40 games each, seed0 500000, custom pruning + hidden-VP value.
* **Result:** depth 1 **0.475 ± 0.155** (371 ms CPU), depth 2 **0.375 ± 0.150** (416 ms),
  depth 3 **0.400 ± 0.152** (490 ms). 0 games over budget in all three.
* **Conclusion:** no evidence that depth helps; all three overlap. CPU grows only modestly with
  depth because the budget stops the deepening.

## E6. Hand-calibrated weights (fail fast)

* **Hypothesis:** F's weights are mis-scaled: `public_vps = 3e14` against `production = 1e8`
  and `hand_resources = 1` makes resources and production worth ~0 in VP units. Re-calibrating
  (1 VP = 1.0, 1 production unit = 0.15 VP, 1 card = 0.06 VP, ...) should play better Catan.
* **Setup:** harness, 40 games each, seed0 500000; calibrated weights file with depth 1 and depth 3.
* **Result:** depth 1 **0.325 ± 0.145**, depth 3 **0.300 ± 0.142**. Clear regression.
* **Conclusion:** the naive recalibration is wrong for this engine/matchup; VP greed is strong.
  This is exactly the kind of question the automated tuner (E10) is for, not hand-tuning. Default
  weights stay at `DEFAULT_WEIGHTS`.

## E7. K worlds (multi-world sampling)

* **Hypothesis:** averaging the one-step evaluation over K sampled worlds reduces hidden-information
  noise and beats F, which samples one world.
* **Setup:** harness, 40 games each, seed0 500000, depth 1, custom pruning + hidden-VP value.
* **Result:** K = 2: **0.125 ± 0.103**, K = 3: **0.125 ± 0.103**. Catastrophic.
* **Conclusion:** a bug, not a modelling result → E8.

## E8. Found and fixed: a truncated depth-1 pass is catastrophic

* **Diagnosis:** the depth-1 loop checked `time.process_time()` and **broke out of the loop** when
  the per-decision budget was exhausted. With K = 2 the pass is twice as expensive, so it was cut
  after the first few actions and the agent returned the best of an arbitrary prefix of the action
  list (often `END_TURN` or a trade) instead of F's move. With K = 1 the same thing happened
  occasionally, which also explains part of E2/E5's regressions.
* **Fix:** the depth-1 pass is now **always completed**; only the optional deepening is
  deadline-limited. Depth 1 costs about what F costs and the allowance is ~6× F's budget, so this
  is always affordable.
* **Re-test (harness, 40 games each, seed0 500000):**
  * K = 1, depth 1: **0.450 ± 0.154**, 349.5 ms CPU, 770 ms worst.
  * K = 2, depth 1: **0.500 ± 0.155**, 573.8 ms CPU, 1340 ms worst.
  * K = 1, depth 3: **0.525 ± 0.155**, 481.8 ms CPU, 970 ms worst.
  * 0 games over budget, 0 errors, 0 timeouts in all three.
* **Conclusion:** the fix removed the large regression. K = 2 and depth 3 are both at least at F's
  level, but 40 games cannot separate them (CI ± 0.15). K = 1 + depth 3 is the default: it has the
  best observed rate, the lower CPU, and leaves K as a knob.

## E9. Final packaged submission

* **Setup:** `bash my_team/make_submission.sh`; unzip `my_team.zip` into a clean directory;
  evaluate the unzipped `agent.py` against F, 10 games, seed0 500000; and against random, 10 games.
* **Result:** zip contains exactly `agent.py`, `search.py`, `weights.json` at the top level.
  Packaged agent vs F: **0.500** (5/10), 525 ms CPU, 850 ms worst, 0 over budget, 0 errors,
  0 timeouts. vs random: **1.000** (10/10), 560 ms CPU, 0 errors.
* **Conclusion:** the submission is self-contained and runs correctly from the zip.

## E10. Tuner smoke test

* **Setup:** `python my_team/tune_weights.py --smoke --workers 2`
  (2 generations, pop 4, 10 games per candidate, v1 as the tuned agent, F as the opponent).
* **First attempt:** every worker returned fitness 0.0 — `multiprocessing.Pool` workers are
  daemonic and `catan_kit.match` must spawn agent processes
  (`AssertionError: daemonic processes are not allowed to have children`). Switched the tuner to
  `concurrent.futures.ProcessPoolExecutor` (non-daemonic workers).
* **Second attempt:** works. Generation 0 best 0.400, generation 1 best 0.500 (10-game noise),
  `tuning_log.csv` and `weights_tuned.json` written, ~32 s wall clock with 2 workers.
* **Conclusion:** the tuning path is exercised end-to-end. The real run is left to the team
  (see README for the command and recommended sizes).

## E11. Tuner scale note (analysis, no run)

* The weights span ~14 orders of magnitude. Multiplicative ES with `sigma = 0.4` moves a weight by
  `exp(0.4) ≈ 1.5` per generation at best, so reaching a 1e4 change takes ~30 generations. If the
  team wants to test a structurally different scale (e.g. a larger `hand_resources`), start the
  tuner from a hand-written `--init` file instead of waiting for the random walk. Flagged for the
  real tuning run.

---

## Summary table (harness only, seed0 = 500000)

| Experiment | Agent A | Games | Win rate vs F | 95% CI | Mean CPU | Worst CPU | Over budget | Errors |
|---|---|---|---|---|---|---|---|---|
| E1 | v1 (safe F) | 20 | 0.500 | ±0.219 | 385 ms | 880 ms | 0 | 0 |
| E2 | v2 first cut | 40 | 0.400 | ±0.152 | 613 ms | 1120 ms | 0 | 0 |
| E2 | v2 first cut vs random | 10 | 1.000 | ±0.000 | 732 ms | 1290 ms | 0 | 0 |
| E4 | custom prune, depth 3 | 40 | 0.375 | ±0.150 | 518 ms | 1020 ms | 0 | 0 |
| E5 | depth 1 | 40 | 0.475 | ±0.155 | 371 ms | 690 ms | 0 | 0 |
| E5 | depth 2 | 40 | 0.375 | ±0.150 | 416 ms | 800 ms | 0 | 0 |
| E5 | depth 3 | 40 | 0.400 | ±0.152 | 490 ms | 1140 ms | 0 | 0 |
| E6 | calibrated weights, depth 1 | 40 | 0.325 | ±0.145 | 317 ms | 760 ms | 0 | 0 |
| E6 | calibrated weights, depth 3 | 40 | 0.300 | ±0.142 | 494 ms | 980 ms | 0 | 0 |
| E7 | K=2, depth 1 (bug) | 40 | 0.125 | ±0.103 | 401 ms | 1000 ms | 0 | 0 |
| E7 | K=3, depth 1 (bug) | 40 | 0.125 | ±0.103 | 504 ms | 1120 ms | 0 | 0 |
| E8 | K=1, depth 1 (fixed) | 40 | 0.450 | ±0.154 | 350 ms | 770 ms | 0 | 0 |
| E8 | K=2, depth 1 (fixed) | 40 | 0.500 | ±0.155 | 574 ms | 1340 ms | 0 | 0 |
| **E8** | **K=1, depth 3 (shipped)** | **40** | **0.525** | **±0.155** | **482 ms** | **970 ms** | **0** | **0** |
| E9 | packaged zip vs F | 10 | 0.500 | ±0.310 | 525 ms | 850 ms | 0 | 0 |
| E9 | packaged zip vs random | 10 | 1.000 | ±0.000 | 560 ms | 800 ms | 0 | 0 |
