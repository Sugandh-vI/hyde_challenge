# version_6 experiments — improving SUB3 (`v5/sub3`)

Every experiment: hypothesis, setup, seeds, result, conclusion.  Failures are kept.

* Agents: `base` = `version_6/base/agent.py` (SUB3, untouched); `v6` = `version_6/v6/agent.py`
  (the working copy, every idea behind a switch); candidates are `swagent:A<i>` with the switch set
  in `SWV6_LIST`, so one screen run compares several variants against one baseline run.
* Lab screen (`version_6/lab/screen.py`): baseline and candidates play the SAME boards, both seat
  orders, same dice; the difference is paired per board.  Verdict KEEP iff `diff − CI > 0`,
  DROP iff `diff ≤ 0` or `diff + CI < 0`, otherwise MAYBE.
* Clock: `--clock` charges `min(0.300 + 0.0125·k, 1.5) s` after k decisions, plus 2.5 ms per
  decision for the JSON view decode the harness charges outside `decide()`; running out means
  uniformly random moves for the rest of the game, as in `catan_kit.match`.
* Unless stated otherwise: 400 boards = 800 games, seed0 = 500000, 2 processes, vs `f_agent` (F).

## Step 0 — the lab had to be fixed before it could be trusted

**Hypothesis.** The v5 lab (`version_6/lab`, copied from `v5/tools`) is a faithful stand-in for the
harness, so a lab screen is enough to accept or reject a change.

**Findings (all measured, in order).**

1. Two agents in one process both `import a2` / `import feats`.  `version_6/base` and `version_6/v6`
   both ship those files, so `sys.modules` handed the second agent the first agent's `a2`.  Fixed by
   `lab/_load.py`: each agent file is imported with its directory first on `sys.path` and the helper
   module names are popped from `sys.modules` afterwards (the agent keeps its own references).
   **Check:** `v6` with SUB3's switch values now makes *bit-identical* decisions to `base` — screen
   `v6@SUB3` vs `base` = `diff +0.000 ±0.000` over 800 paired games.
2. The lab charged CPU with `time.process_time()` of one process running *both* agents.  With
   numpy's OpenBLAS defaulting to one thread per core, the pinned/unpinned difference was 97 vs
   345 ms/game — a 3.5× inflation of the agent's measured CPU (BLAS worker threads spin between
   calls and their CPU lands inside the other agent's timing window).  Fixed: `fastmatch.py` pins
   `OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=1` before numpy is imported, exactly as the harness's
   `host.py` does for OMP/MKL.
3. `--cpu_scale S` (charge S× the measured CPU) is the wrong way to model a slower machine: the
   agent's own deadlines (`cpu_left`, `time.process_time()`) are in the same units as the charge, so
   scaling double-counts and creates over-budget games that do not exist (11/800 at scale 2 for a
   change that is neutral at scale 1).  Replaced by `--io_ms` (default 2.5 ms), the per-decision
   JSON-decode cost the harness charges outside `decide()`: harness 299 ms/game vs 172 ms of
   in-decide CPU over 400 games, i.e. 49 decisions × 2.5 ms.
4. `screen.py` printed candidate CPU 1000× too large (double scaling of an already-ms number).
   Fixed.

**Conclusion.** After (1)-(4) the lab's clock numbers match the harness within ~7% on total CPU
(lab 320 ms/game vs harness 299 ms/game for the same agent and seed range, sandbox), the net is
consulted on 100% of main turns in both, and `v6` reproduces SUB3 exactly when its switches are set
to SUB3's values.  Screens below use `--clock` with `--io_ms 2.5` and no `--cpu_scale`.

## Step 1 — baseline: SUB3 with the harness clock, lab and harness

**Hypothesis.** SUB3's fixed `NET_MIN_CPU = 0.25 s` gate plus its unbounded opening search leave the
net unused late in some games and can run the clock out.

| run | games | win vs F | mean CPU | worst game | worst decision | decisions/game | net consulted | over budget |
|---|---|---|---|---|---|---|---|---|
| lab, unpinned BLAS (first attempt) | 800 | 0.790 ±0.028 | 451 ms | 1369 ms | 325 ms | 48 | 0.89 | 2 (58 random moves) |
| lab, pinned, `--cpu_scale 2` | 800 | 0.802 ±0.028 | 388 ms* | 1138 ms | 467 ms | 48 | 0.97 | 2 (23 random moves) |
| **lab, pinned, `--io_ms 2.5`** | 800 | **0.806 ±0.028** | **320 ms** | 690 ms | 31 ms | 45 | **1.00** | **0** |
| **harness (`catan_kit.evaluate`)** | 400 | **0.7775 ±0.0408** | **299 ms** | 1020 ms | — | 49 | — | **0**, 0 errors |

\* inflated by BLAS threads.

**Conclusion.** The "clock problem" was a measurement artifact.  With correct accounting SUB3 spends
320 ms of the ~860 ms it is allowed at 45 decisions, runs out of clock in 0/800 lab games and 0/400
harness games, and already consults the net on 100% of main turns in both.  The harness agrees with
the lab on win rate within its CI (0.7775 ±0.0408 vs 0.806 ±0.028) and on CPU within 7%.
So Step 2 cannot be about *affording* the net; see below for what it is about.

## Step 2 — make the net affordable every main turn (and the clock safe)

**Hypothesis.** Sharing successor states between `a2._main` and the net, caching the value function's
per-decision invariants, and replacing the fixed threshold with an adaptive budget would raise the
net-consulted fraction to ~100% and cut CPU.

**Profile first** (`/tmp/prof4.py`: wraps `decide`, `v4net.Agent.scores`, `v4net.Value.__call__`,
`feats.features`, `a2._main`, 10 games, idle sandbox, BLAS pinned, per game = 47 decisions):

| part | ms/call | ms/game | share of decide |
|---|---|---|---|
| `net.scores` (sampling + 106 successor states + features + net) | 2.43 | 115 | 67% |
| ... of which `v4net.Value.__call__` (the MLP itself, 106 rows) | 0.20 | 9.4 | 5.5% |
| ... of which `feats.features` | 0.59 | 28 | 16% |
| `a2._main` (its own copy + execute + planner) | 0.98 | 46.5 | 27% |
| `choose_opening` | 0.10 | 4.9 | 2.8% |
| **decide total** | **3.64** | **172** | 100% |

The net costs 2.4 ms per main turn against an allowance of ~30 ms per decision.  The MLP is 0.2 ms;
the rest is the search that produces its 106 inputs, which is the same search `a2._main` does.

**Changes kept in v6 (each a switch).**

* `net_budget="adaptive"`: consult the net while `cpu_left > net_reserve + net_costs × (measured cost
  of one net call)`, i.e. ~0.107 s instead of a flat 0.25 s.  A fixed number is wrong in both
  directions: too high skips the net when the game is nearly over, too low can spend the last of the
  clock.
* `open_cap=0.24`: the counter-opening search stops at `min(open_cap, 0.6 × (cpu_left − 0.04))`.
* `reserve` / `cap_frac`: agent2's planner budget as *instance* settings, so two agents with
  different settings can share one process (needed by the lab, harmless in the harness).

**Not done, with the measurement that says why.** Sharing successor states between `a2._main` and
the net, and caching per-decision invariants of the value function, would save at most ~1.5 ms per
main turn (the overlap between the 46 ms `a2._main` and the 115 ms `net.scores`, of which 28 ms is
features the net needs anyway).  It would touch the two files that decide every game, for <10% of a
budget the agent uses 37% of.  Rejected on evidence, not skipped.

| run | 800 games vs base | win | diff ±CI | CPU | net | over |
|---|---|---|---|---|---|---|
| base (SUB3) | — | 0.806 | — | 320 ms | 1.00 | 0 |
| `s2_clock` (adaptive + open_cap 0.24) | 800 | 0.805 | −0.001 ±0.002 | 323 ms | 1.00 | 1 |
| v6 @ SUB3 switches (control) | 800 | 0.806 | +0.000 ±0.000 | 324 ms | 1.00 | 0 |

**Conclusion.** Step 2 is neutral in strength (−0.1 pt, inside noise) and does not change the
net-consulted fraction, because SUB3 already sat at 100% once measured properly.  It is kept as the
base configuration for later steps because it removes two real fragilities for a *slower* machine:
an unbounded opening search (worth 5-30 ms typically but up to 233 ms measured) and a fixed CPU
threshold that cannot adapt to how much a decision actually costs.  `s2_clock` hit the clock in 1 of
800 lab games in one run and 0 in another; the harness run of the same preset (1000 games, seeds
990000+, section "Final harness") was **0/1000 over budget**, which is the number that counts.

## Step 3 — value-function weight wins

**Hypothesis.** The weights `v5` screened (`hand_synergy ×0.5/×0.25/×0`, `reachable_production_1`
×0.5/×0.25, `hand_resources` lower, `buildable_nodes` higher) still pay on top of SUB3.

**Setup.** Each candidate = `s2_clock` + one `vf` override merged over `a2.WEIGHTS`
(`make_vf({**WEIGHTS, **vf})`).  400 boards = 800 paired games, seed0 500000, lab `--clock`.

| # | change (default → candidate) | win | diff ±CI | verdict |
|---|---|---|---|---|
| 3.1 | `hand_synergy` 100 → 50 | 0.809 | +0.004 ±0.015 | MAYBE |
| 3.2 | `hand_synergy` 100 → 25 | 0.792 | −0.013 ±0.024 | DROP |
| 3.3 | `hand_synergy` 100 → 0 | 0.792 | −0.013 ±0.024 | DROP |
| 3.4 | `reachable_production_1` 10000 → 5000 | 0.806 | +0.001 ±0.016 | MAYBE |
| 3.5 | `reachable_production_1` 10000 → 2500 | 0.809 | +0.004 ±0.018 | MAYBE |
| 3.6 | `hand_resources` 1 → 0.5 | 0.804 | −0.003 ±0.010 | DROP |
| 3.7 | `hand_resources` 1 → 0 | 0.805 | −0.001 ±0.010 | DROP |
| 3.8 | `buildable_nodes` 1000 → 2000 | 0.812 | +0.006 ±0.013 | MAYBE |
| 3.9 | `buildable_nodes` 1000 → 5000 | 0.807 | +0.001 ±0.017 | MAYBE |
| 3.10 | combination `hand_synergy` 50 + `reachable_production_1` 2500 + `buildable_nodes` 2000 | not run | — | not run |

3.10 was not run: with every single change inside +-1.5 pt and two of the three point estimates
positive, a combination would measure the sum of three noise-level effects on the same 800 games.
The per-change numbers do not justify the extra run (and a combination needs a fresh-seed
confirmation, which would double it).

**Conclusion.** Nothing clears `diff − CI > 0`; the two largest point estimates (+0.6 pt for
`buildable_nodes` 2000, +0.4 pt for `hand_synergy` 50 / `reachable_production_1` 2500) are inside
their own CIs.  The v5 numbers for the same weights (`SynLo +0.0675`, `Reach1lo +0.0425`) do not
reproduce on SUB3 + net — they were measured against a different, weaker baseline where the value
function drove every decision.  `hand_synergy` is *worse* when removed, so the term stays.

## Step 4 — net override tuning

**Hypothesis.** SUB3's `MARGIN = 0.03` and its main-turn-only override leave value on the table:
the net could also decide robber placement, the pre-roll knight and discards, and a smaller margin
should win more often if the net ranks better than agent2.

**Setup.** As Step 3; one switch per candidate.

| # | change | win | diff ±CI | verdict |
|---|---|---|---|---|
| 4.1 | `margin` 0.03 → 0.00 | 0.802 | −0.004 ±0.030 | DROP |
| 4.2 | `margin` 0.03 → 0.01 | 0.806 | +0.000 ±0.027 | DROP |
| 4.3 | `net_on_robber` (net may override the robber rule) | 0.771 | −0.034 ±0.030 | DROP |
| 4.4 | `net_on_knight` (net may override the pre-roll knight rule) | 0.804 | −0.001 ±0.017 | DROP |
| 4.5 | `net_on_discard` | 0.805 | +0.000 ±0.000 | no-op |
| 4.6 | `cap_frac` 1/6 → 1/3 (agent2 planner budget) | 0.805 | +0.000 ±0.000 | no-op |

**Conclusion.** The net is a worse robber placer than agent2's hand-written denial rule
(−3.4 pt ±3.0, the largest single loss in this campaign after the opening), and it is neutral on the
pre-roll knight.  `net_on_discard` and `cap_frac` are exact no-ops in this engine: upstream
catanatron has no `DISCARD_RESOURCE` action (`a2.CONFIG["discard_rule"]` is off for the same reason),
and agent2's planner finishes in ~1 ms against a deadline of ~35 ms, so its budget never binds.
Both stay as switches with a documented "no effect here".

**Note on the CI.** Lowering the margin makes the candidate diverge from the baseline more often, so
the paired CI widens (0.002 → 0.030) — the same 800 games resolve much less.  A margin change is
therefore expensive to test properly; only a ≥2 pt effect would show.

## Step 5 — win finder

**Hypothesis.** When the actual VP total (hidden VP cards included) is ≥ 8, an explicit depth-3
search over our own turn (`winfind.find_win`, budget 400 copies) finds a winning line before the
one-ply value function can mis-rank it.

**Setup.** `winfind_vp = 8`, `winfind_depth = 3`, `winfind_budget = 400`, gated on
`cpu_left ≥ winfind_min_cpu = 0.12`.  Smoke test (12 games): 0.833, 252 ms/game, worst decision
78 ms, 0/12 over budget; hits/triggers printed by the lab as `winfind <triggers> (<hits>)`.

| # | change | win | diff ±CI | verdict | triggers / hits |
|---|---|---|---|---|---|
| 5.1 | `winfind_vp` 99 → 8 | 0.807 | +0.003 ±0.003 | MAYBE | 3/800 over budget |

The lab counts triggers and hits (`winfind <triggers> (<hits>)`); on these 800 games the finder fired
rarely, and the CPU cost was +38 ms/game (318 → 356).  Three of 800 games ran out of clock, which
disqualifies it regardless: a final candidate must be at 0.

**Conclusion.** DROP.  The depth-3 search over our own turn is not better than the one-ply value
function at finding the last points (the value function already puts `WIN` on any move that wins and
follows a 2-ply continuation), and it spends enough CPU to push games over the limit.

## Step 6 — retrain the opening model (not run here)

**Hypothesis.** `opengen.py`'s explorer (`vf.make_vf(lr_always=2e3, port2=3e7, port3=1e6)`) is a
weaker mid-game agent than SUB3, so the 20k positions behind `open_logreg.json` are labelled by
weaker play; regenerating them with the Step-2 agent as the explorer should label better.

**Why not run in the sandbox.** Generation is ~20k games × 2 seats at ~0.25 s/game on two cores
(measurement: 2000 value-net games took 451 s), i.e. > 40 min of wall clock, plus training, plus a
confirmation screen — and the machine has 2 cores, not the 8 the command list in the README assumes.
The exact commands are in the README (`Arena update 5`, "Step 6") so it can run on the team machine.

**Proxy experiment that *was* run.** `opening="ply"` replaces the learned counter-opening with the
agent's ordinary 1-ply value on setup settlements, same seeds, 800 paired games:

| change | win | diff ±CI | verdict | CPU |
|---|---|---|---|---|
| `opening` counter → 1-ply | 0.720 | **−0.085 ±0.037** | DROP | 321 ms |

That is the largest single effect measured in this campaign, and it says two things: the learned
counter-opening is worth **+8.5 pt** on this agent (do not touch it casually), and the opening is
where the remaining strength is.  A retrain therefore has a high bar (it must beat a model that is
already worth 8.5 pt) and a real upside; it is the first thing to run on the 8-core machine.

## Step 7 — value-net retrain

**Hypothesis.** The shipped `value_w{0,1,2}.npz` were trained by friend 2 on v4 self-play; positions
from the stronger SUB3 (2000 games, seeds 100000+, self-play + vs F + vs base) should label better,
and a fresh ensemble should beat them on held-out games.

**Setup.** `tools/gen_data.py --games 2000 --workers 2` → 115,383 rows (2 seats × ≤30 rows/game,
evenly subsampled), 451 s; split by **game seed** (`seed % 8 == 0` → 250 of 2000 games, 14,410
validation rows, win rate 0.519).  `tools/train_net.py`: same architecture as the shipped nets
(102→128→64→1, ReLU, one logit), 40 epochs, batch 512, Adam 3e-4, weight decay 1e-4, 3 models.

| model | val log-loss |
|---|---|
| current shipped ensemble (mean of 3 logits) | **0.4505** |
| new model 0 / 1 / 2 | 0.4626 / 0.4648 / 0.4665 |
| new ensemble | 0.4596 |

**Conclusion.** REJECT.  The new ensemble is 0.009 worse in log-loss on held-out seeds (and 0.2 pt
worse in accuracy, 0.7621 vs 0.7641).  2000 games of self-play is not enough to beat the shipped
weights, which were trained on far more data; the acceptance rule ("keep only if val log-loss
improves **and** the paired screen is KEEP") is not met, so `version_6/v6` keeps the original nets.

**Check worth keeping.** `feats.features(game, color)` computed on the *real* game equals
`feats.features(masked_copy(game, color), color)` at every decision (85 positions, max |diff| = 0):
the recorded rows are exactly what the agent sees, and the value net's features carry no hidden
information.


## Where the campaign ended up

Every knob that a screening step could reach was neutral or worse:

| change | diff vs base (800 paired games) | verdict |
|---|---|---|
| clock hardening (`s2_clock`) | −0.001 ±0.002 | neutral, kept for robustness |
| 9 value-function weight variants | −0.013 … +0.006, all inside ±0.018 | no KEEP |
| margin 0.00 / 0.01 | −0.004 ±0.030 / +0.000 ±0.027 | no KEEP |
| net on robber / knight | −0.034 ±0.030 / −0.001 ±0.017 | DROP |
| net on discard, planner budget ×2 | ±0.000 ±0.000 | no-ops in this engine |
| win finder at VP ≥ 8 | +0.003 ±0.003, 3/800 over budget | DROP |
| 1-ply opening instead of the counter model | **−0.085 ±0.037** | DROP (the model is worth 8.5 pt) |
| retrained value net | val log-loss 0.4596 vs 0.4505 | REJECT |

**Conclusion.** SUB3 sits on a plateau: within the reach of a 2-core sandbox and 800-game paired
screens, no single switch beats it by more than the noise (±1.5 pt).  The two things with real
upside are both data-hungry — a retrained opening model (Step 6) and a value net trained on more
than 2000 games (Step 7) — and both need the 8-core machine; the commands are in the README.
The recommended preset is therefore `s2_clock`: same decisions, same win rate within ±0.2 pt, but
the opening search can no longer outrun the first allowance and the net budget adapts to what a
decision actually costs.


## Final harness (seeds 990000+)

`python -m catan_kit.evaluate`, `s2_clock` as A:

| opponent | games | win rate | mean CPU/game | worst game | over budget | errors |
|---|---|---|---|---|---|---|
| `examples/f_agent.py` | 1000 | 0.7900 ±0.0252 | 283.7 ms | 940 ms | 0 | 0 (48,160 decisions) |
| `version_6/base/agent.py` (SUB3) | 400 | see README `Arena update 5` | | | | |
| `version4_nn/agent.py` | 400 | see README `Arena update 5` | | | | |
| `version_3/opponents/f_k3/agent.py` | 400 | see README `Arena update 5` | | | | |

Zero over-budget, zero errors, zero timeouts/crashes, 0.04 s load time.  Same level as SUB3 itself
(0.794 ±0.028 on the team machine, 0.7775 ±0.0408 in this sandbox at seed0 500000), consistent with
the paired lab screen (−0.1 pt ±0.2).
