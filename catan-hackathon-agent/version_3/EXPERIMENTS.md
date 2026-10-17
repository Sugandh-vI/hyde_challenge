# version_3 — experiments

Hypothesis, setup, seeds, result, conclusion. Every harness number below was produced by
`catan_kit.evaluate.run` (the scoring script), through `version_3/tools/run_pool.py` or
`version_3/tools/compare.py`, and is also in `version_3/results.csv`.

Seeds: development `seed0 = 500000` (all runs below), tuning `100000+` (sweep), final checks
`990000+` (left for the team). Seats alternate and each board is played twice, once per seat.

CPU caveat: this development sandbox is ~2.2x slower than the scoring machine on the kit's own
benchmark (F costs 350 ms/game here, 158 ms on the machine). Every CPU number below should be
divided by ~2.2 before comparing with the 450 ms target.

---

## E0. Environment

* **Setup:** fresh Python 3.11.2 venv, `pip install -r requirements.txt` (+ CPU torch).
  `python -m catan_kit.selfcheck`.
* **Result:** all checks passed; engine determinism reproduced (`seed 4256 -> ('Color.RED', 111)`).
  F vs random 20/20 at 350 ms CPU/game here.
* **Conclusion:** the numbers below are trustworthy and comparable to the kit's own benchmark.

## E1. Value function: zero extras must equal F exactly

* **Hypothesis:** re-implementing `base_fn` with our extra terms switched off reproduces F
  bit-for-bit.
* **Setup:** `python version_3/tools/check_value.py`; 242 successor positions from random-agent
  games, both colours.
* **Result:** max relative difference **0.000e+00**; `base_fn` 0.105 ms/call, `v3_value`
  0.058 ms/call. With the shipped extras on, the value moves by up to 0.5 (relative).
* **Conclusion:** the extra-term implementation cannot silently break F's ordering, and the
  value function is not a CPU problem. This test is the guard that makes the extras safe to tune.

## E2. Opening opponent model ("f_value") predicts F's setup picks

* **Hypothesis:** during the setup phase F is exactly a 1-ply argmax of `base_fn`, so our model
  can predict its placements; restricting the model to the top-`k` candidates by a pip proxy
  should be safe.
* **Setup:** `python version_3/tools/check_fmodel.py`; 24 F-vs-F games, 96 recorded settlement
  decisions, the model evaluated on each.
* **Result:** exact agreement **96/96 (1.000)** with a full scan and with `opp_max_eval = 12`;
  95/96 at 10; 94/96 at 8; 89/96 at 6.
* **Conclusion:** use `opp_max_eval = 12` (exact in this sample, 25 % cheaper than 16). This makes
  the counter-opening a model of the *real* F, not a guess.

## E3. Preset screen, 40 games each vs F (`seed0 = 500000`)

* **Hypothesis:** the opening search is the big lever; rules and search add a smaller amount.
* **Setup:** `run_pool.py --pool examples/f_agent.py --games 40 --workers 2`, one run per preset.
  P0/P6/P7 are diagnostics, not part of the five presets.
  * P0 = value extras only (no opening search, no rules, no search)
  * P1 = opening(f_value) + value extras
  * P2 = P1 + RULES  (== P7)
  * P3 = P2 + SEARCH (default)
  * P4 = P3 with the "mix" opening model
  * P5 = P3 with the "generic" opening model
  * P6 = opening + search, no rules
* **Result (win rate ± 95 % CI, mean CPU/game, worst CPU, over budget, errors):**

  | Preset | Win rate | CPU mean | CPU worst | Over | Errors |
  |---|---|---|---|---|---|
  | P0 (no opening) | 0.450 ±0.154 | 286 ms | 670 ms | 0 | 0 |
  | P1 | **0.650 ±0.148** | 425 ms | 770 ms | 0 | 0 |
  | P2 (= P7) | 0.500 ±0.155 | 351 ms | 720 ms | 0 | 0 |
  | P3 | 0.575 ±0.153 | 406 ms | 680 ms | 0 | 0 |
  | P4 | **0.650 ±0.148** | 440 ms | 820 ms | 0 | 0 |
  | P5 | 0.600 ±0.152 | 342 ms | 520 ms | 0 | 0 |
  | P6 | 0.575 ±0.153 | 506 ms | 1080 ms | 0 | 0 |

* **Conclusion:** every preset with the counter-opening is at or above F, and the difference
  between having it and not having it (P1 0.650 vs P0 0.450) is the largest effect we measured.
  At 40 games the CI is ±0.15, so the ordering among P1–P5 is *not* resolved by this screen; it
  needs the paired runs (E4, E5).

## E4. Paired comparison P1 vs P3, 80 games

* **Hypothesis:** rules + search beat the pure opening preset.
* **Setup:** `compare.py --a p1 --b p3 --opponent examples/f_agent.py --games 80 --seed0 500000`
  (identical boards and dice for both agents).
* **Result:** P1 **0.575** vs P3 **0.6625**; paired difference **+0.0875 ±0.1167** in P3's
  favour; discordant games 8 (P1 only) vs 15 (P3 only). P1 459 ms, P3 407 ms mean CPU, both
  0 errors / 0 over budget.
* **Conclusion:** the point estimate favours P3 but the 95 % interval includes 0. Not decisive at
  80 games — exactly the comparison the team should re-run at 400+ games.

## E5. Paired comparison P3 vs P4 (mixed opening model), 80 games

* **Hypothesis:** hedging the opponent model costs nothing against F (F is exactly the "f_value"
  model) and should be safer against non-F opponents.
* **Setup:** `compare.py --a p3 --b p4 --games 80 --seed0 500000`.
* **Result:** P3 **0.625** vs P4 **0.6125**; paired difference **+0.0125 ±0.1281** (P3);
  discordant 14 vs 13. P3 407 ms, P4 424 ms mean CPU, 0 errors / 0 over budget.
* **Conclusion:** statistically identical against F, as predicted. The choice between them is a
  robustness question, not a strength question.

## E6. Pool run, 20 games per opponent

* **Hypothesis:** the agent is not F-specific and beats weaker/perturbed opponents.
* **Setup:** `run_pool.py --games 20 --workers 2` with the default pool
  (F, `my_team/agent_v2.py`, random, `f_k3` = F averaged over 3 worlds, `f_eps` = F with 10 %
  random moves).
* **Result:**

  | Opponent | P3 | P4 |
  |---|---|---|
  | F | 0.500 (10/20) | 0.700 (14/20) |
  | my_team/agent_v2 (same-turn search) | 0.400 (8/20) | 0.550 (11/20) |
  | random | 1.000 (20/20) | 1.000 (20/20) |
  | f_k3 (F over 3 worlds) | 0.650 (13/20) | 0.800 (16/20) |
  | f_eps (F + 10 % random) | 0.700 (14/20) | 0.600 (12/20) |

  0 errors, 0 timeouts, 0 games over budget in all 200 games.
* **Conclusion:** both presets beat the random agent 100 %, the multi-world F and the noisy F, and
  lose ground to our own previous search agent (`my_team/agent_v2`) in this small sample. 20-game
  intervals are ±0.22, so only the random result and the "not F-specific" claim are solid. P4
  leads in this sample by an amount that is not significant.

## E7. Sweep smoke test

* **Setup:** `python version_3/tools/sweep.py --smoke` (3 candidates × 8 games × 1 generation,
  2 workers, tuning seeds 100000+).
* **Result:** works end-to-end in 11 s: candidates built in temp folders, evaluated in parallel by
  `ProcessPoolExecutor`, fitness 0.375 / 0.250, log and best-so-far written.
* **Conclusion:** the tuning path is exercised; the real sweep is left to the team (8 cores).

## E8. Packaged submission

* **Setup:** `bash version_3/make_submission.sh version_3/v3`, unzip into a clean directory,
  evaluate the unzipped `agent.py` against F (8 games) and F/K3/random in the pool runs.
* **Result:** the zip holds exactly `agent.py`, `v3agent.py`, `v3_*.py` (5 modules) and
  `weights.json`, `agent.py` at the top level. Unzipped agent vs F: 0.500 (4/8), 386 ms CPU,
  0 over budget, 0 errors, 0 timeouts.
* **Conclusion:** the submission is self-contained and runs from the zip with the same numbers.

## E9. Opening CPU

* **Setup:** in-process timing of the first four decisions over 6 games (both seats), measuring
  `Agent.decide` with `obs.cpu_left = 1.2`.
* **Result:** first settlement **102 ms mean / 163 ms max** (44–73 ms when we are the first
  player, ~161 ms when we are the second player, because the second player's plan needs a reply
  for every (c1, c2) pair); first road 0.3 ms; second settlement 19 ms mean / 40 ms max; second
  road 0.3 ms. Whole game 348 ms mean / 637 ms max.
* **Conclusion:** the opening is within the initial 312 ms allowance with room to spare, and the
  mid-game cap (11 ms/decision) makes the rest of the game cheap. No game exceeded the clock in
  any of the 800+ games run for this report.

---

## What was NOT run (left to the team's 8-core machine)

1. The real sweep (`--pop 24 --gens 6 --games 200`), i.e. actual weight tuning. Only the smoke
   variant was run, as instructed.
2. 400–1000-game evaluations of each preset. All conclusions above are at 40–80 games
   (CI ±0.12–0.15), and the instructions explicitly forbid long runs here.
3. Anything against **O**: we do not have it.
