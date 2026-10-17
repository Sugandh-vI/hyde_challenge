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

---

## Arena update 3 — version_3 (new strategy)

A second, independent agent lives in `catan-hackathon-agent/version_3/`. **`my_team/` is
untouched** and remains a working fallback; everything below is new code. `version_3/v3/` is
self-contained (agent.py + 5 helper modules + weights.json) and is what gets zipped.

### 1. Files created

All under `catan-hackathon-agent/version_3/`.

| File | One line |
|---|---|
| `v3/agent.py` | Submission entry point: `PRESET` constant, imports the core. |
| `v3/v3agent.py` | The Agent: preset switch, opening hook, rules hook, 1-ply/search, budget, fallback chain. |
| `v3/v3_value.py` | F's `base_fn` re-implemented plus the extra terms; weights loader (`weights_best.json` wins over `weights.json`). |
| `v3/v3_openfeat.py` | 21 opening features per side (pips, diversity, numbers, ports, balance, open follow-ups) and the opening score. |
| `v3/v3_opening.py` | Counter-opening: enumerate our picks, predict the opponent's replies (F model), score the final four settlements; plus the initial-road heuristic. |
| `v3/v3_search.py` | Same-turn iterative deepening: depth 1 is always completed, chance nodes probability-weighted by `execute_spectrum`. |
| `v3/v3_rules.py` | MOVE_ROBBER tile choice, DISCARD_RESOURCE choice, endgame denial multiplier. |
| `v3/weights.json` | Base weights (F's 13), extra weights, opening coefficients and search sizes — one tunable file. |
| `v3_presets/p1..p5/agent.py` | Thin wrappers so each configuration can be evaluated directly (also p0/p6/p7 diagnostics). |
| `opponents/f_k3/agent.py` | Pool opponent: F's 1-ply averaged over 3 sampled worlds. |
| `opponents/f_eps/agent.py` | Pool opponent: F with 10 % uniformly random moves. |
| `tools/run_pool.py` | One agent vs a pool, parallel chunks, results logged to `version_3/results.csv`. |
| `tools/compare.py` | Paired comparison of two agents on the same boards with a paired 95 % interval. |
| `tools/sweep.py` | Random search over weights.json, scored by win rate vs F, `ProcessPoolExecutor`, resumable best. |
| `tools/check_value.py` | Unit test: zero extras reproduces F's value exactly. |
| `tools/check_fmodel.py` | Measures how often the opening's F-model predicts F's actual setup picks. |
| `make_submission.sh` | `bash version_3/make_submission.sh version_3/v3` → `version_3/v3.zip` with agent.py at the top. |
| `EXPERIMENTS.md` | E0–E9 with hypothesis, setup, seeds, result, conclusion. |
| `results.csv` | Every harness run (the columns of `my_team/results.csv` plus `opponent`). |

### 2. Design

* **Opening (the biggest lever).** Snake order is first, second, second, first. At each of our
  picks we enumerate our best `k1` candidates, let a model place the opponent's remaining
  settlements, and score the final four settlements with `opening_score(ours) −
  opening_score(theirs)`; we take the node whose *final* difference is best. That is what makes it
  a counter-opening: we choose the spot that is still good after the opponent answers, instead of
  the spot that looks best on its own.
  * The **F model** executes each legal BUILD_SETTLEMENT on a copy and scores it with F's own
    `base_fn`, in engine order, first maximum wins — exactly catanatron's `ValueFunctionPlayer`.
    During setup there is no hidden information, so this is exact: `check_fmodel.py` measures
    **96/96** correct predictions at `opp_max_eval = 12`.
  * `generic` uses our opening score for the opponent, `mix` averages the two models' resulting
    positions (hedge against non-F opponents).
  * The initial road points at the best still-open settlement one step beyond it.
  * CPU: 102 ms mean / 163 ms max for the first settlement, 19 ms for the second, 0.3 ms per road.
* **Value function.** `base_fn` + extras: always-on longest road (1000, a tie-breaker above
  `num_tiles` but far below production), owned 2:1 port weighted by our production of that
  resource (2e7), 3:1 port (1e6), our own hidden VP cards (1.0 × `public_vps` — we can see our
  own development cards, `base_fn` only sees public VPs), and slots for opponent VPs and
  distance-to-next-build that default to 0 and are swept. **With every extra at 0 the value equals
  F's exactly (measured 0.000e+00 relative difference on 242 positions).**
* **Mid-game.** Depth 1 = F's rule over *all* legal actions. The depth-1 pass is always completed:
  cutting it short was measured at 0.125 against F in the previous session. `SEARCH` then adds
  same-turn iterative deepening (depth 3, beam 3, root beam 5), stopping lines at END_TURN or a
  change of turn, with chance nodes probability-weighted (`execute_spectrum`; robber steals use
  the true card-count probabilities, which are public in 2-player Catan). `list_prunned_actions`
  is not used: its robber pruning cost ~10 points of win rate in the previous session.
* **Rules.** Robber: opponent pips on the tile minus ours, plus a steal bonus and a bonus for
  denying the resource they are short of for their nearest build. Discard: keep the cards closest
  to our nearest build (city > settlement > dev > road). Endgame: at 8+ opponent public VPs the
  enemy-production weight is multiplied by 2.5 and the robber doubles the denial bonus.
* **Clock.** `budget = min(cap, (cpu_left − 140 ms) / 38)` with cap 11 ms mid-game, 160 ms for the
  first settlement and 80 ms for the second; every search loop checks `time.process_time()` and
  returns its best completed level. Below 3 ms the agent ends its turn instead of thinking.
* **Fallbacks.** full policy → plain 1-ply with F's `base_fn` → first ROLL/END_TURN → index 0.

### 3. Exact commands (from `catan-hackathon-agent/`)

```bash
cd catan-hackathon-agent
source .venv/bin/activate            # on the team machine

# 0. install check
python -m catan_kit.selfcheck

# 1. unit test (value == F with zero extras) and the opening model check
python version_3/tools/check_value.py
python version_3/tools/check_fmodel.py

# 2. each preset vs F, 400 games each (~2-4 min each on 8 cores)
for p in p1 p2 p3 p4 p5; do
  python version_3/tools/run_pool.py --agent version_3/v3_presets/$p/agent.py \
      --pool examples/f_agent.py --games 400 --seed0 500000 --workers 7 --tag ${p}_vs_F_400
done

# 3. the shipped agent vs the full pool (F, my_team/agent_v2, random, f_k3, f_eps), 400 each
python version_3/tools/run_pool.py --agent version_3/v3/agent.py \
    --games 400 --seed0 500000 --workers 7 --tag v3_pool_400

# 4. paired comparison of the two best presets (much tighter interval than 400 games alone)
python version_3/tools/compare.py --a version_3/v3_presets/p3/agent.py \
    --b version_3/v3_presets/p4/agent.py --opponent examples/f_agent.py \
    --games 400 --seed0 500000

# 5. full weight sweep (8 cores; ~5 min per generation at these sizes)
python version_3/tools/sweep.py --pop 24 --gens 6 --games 200 --workers 7 --seed0 100000
#    the best is written to version_3/v3/weights_best.json and loaded automatically;
#    delete that file to fall back to version_3/v3/weights.json.
#    after a sweep: python version_3/tools/run_pool.py --agent version_3/v3/agent.py \
#        --pool examples/f_agent.py --games 1000 --seed0 990000 --workers 7 --tag tuned_final

# 6. build the submission zip (agent.py at the top level)
bash version_3/make_submission.sh version_3/v3
#    -> version_3/v3.zip ; upload it with  ~/upload.sh version_3/v3.zip

# switching the shipped preset: edit PRESET in version_3/v3/v3agent.py (default "P3"),
# or zip a preset folder:  bash version_3/make_submission.sh version_3/v3_presets/p4
```

### 4. Test results actually obtained

Seeds `seed0 = 500000`, seats alternated, sandbox numbers (F itself costs 350 ms/game here vs
158 ms on the scoring machine). Full detail in `version_3/EXPERIMENTS.md`, rows in
`version_3/results.csv`.

Unit test (`tools/check_value.py`):

```
positions           242
max relative diff   0.000e+00   (must be 0)
base_fn             0.105 ms/call
v3_value            0.058 ms/call
shipped extras move the value by up to 0.5000 (relative)
OK
```

Opening model (`tools/check_fmodel.py`, 24 F-vs-F games, 96 settlement decisions):

```
opp_max_eval=None: exact agreement 96/96 = 1.000
opp_max_eval=  12: exact agreement 96/96 = 1.000
opp_max_eval=  10: exact agreement 95/96 = 0.990
opp_max_eval=   6: exact agreement 89/96 = 0.927
```

Presets vs F, 40 games each (`run_pool.py`, one row per preset):

```
p0  (no opening)   0.450 +-0.154  cpu  286.0 ms worst  670.0  over 0 err 0
p1                 0.650 +-0.148  cpu  425.2 ms worst  770.0  over 0 err 0
p2 (=p1+rules)     0.500 +-0.155  cpu  351.0 ms worst  720.0  over 0 err 0
p3 (default)       0.575 +-0.153  cpu  406.2 ms worst  680.0  over 0 err 0
p4 (mix opening)   0.650 +-0.148  cpu  439.8 ms worst  820.0  over 0 err 0
p5 (generic open)  0.600 +-0.152  cpu  342.0 ms worst  520.0  over 0 err 0
p6 (p1+search)     0.575 +-0.153  cpu  506.2 ms worst 1080.0  over 0 err 0
```

Paired runs (same boards and dice for both agents, so the interval is much tighter):

```
P1 vs P3, 80 games:  A 0.5750 (46/80)   B 0.6625 (53/80)
                     A - B -0.0875 +-0.1167   discordant 8 / 15
                     A cpu 458.8 ms, B cpu 406.9 ms, 0 errors, 0 over budget

P3 vs P4, 80 games:  A 0.6250 (50/80)   B 0.6125 (49/80)
                     A - B +0.0125 +-0.1281   discordant 14 / 13
                     A cpu 406.9 ms, B cpu 423.6 ms, 0 errors, 0 over budget
```

Pool, 20 games per opponent (`run_pool.py`, full default pool):

```
p3: f_agent      0.500 (10/20)  cpu 460.0 ms  over 0 err 0
    agent_v2     0.400 ( 8/20)  cpu 357.5 ms  over 0 err 0
    random_agent 1.000 (20/20)  cpu 425.0 ms  over 0 err 0
    f_k3         0.650 (13/20)  cpu 412.5 ms  over 0 err 0
    f_eps        0.700 (14/20)  cpu 398.0 ms  over 0 err 0

p4: f_agent      0.700 (14/20)  cpu 436.0 ms  over 0 err 0
    agent_v2     0.550 (11/20)  cpu 416.0 ms  over 0 err 0
    random_agent 1.000 (20/20)  cpu 427.5 ms  over 0 err 0
    f_k3         0.800 (16/20)  cpu 441.5 ms  over 0 err 0
    f_eps        0.600 (12/20)  cpu 404.0 ms  over 0 err 0
```

Packaged zip (unzipped into a clean directory, 8 games vs F):

```
A = /tmp/zv3/agent.py         B = examples/f_agent.py
A win rate          0.5000 +-0.3465   (4/8)
A mean CPU / game      386.2 ms
A worst CPU / game     540.0 ms
A games over budget        0      (random moves played after the limit: 0)
A errors                   0      illegal index or exception, of 436 decisions
A timeouts/crashes    0 / 0
A load time             0.01 s
```

Opening CPU (in-process, 12 seats): first settlement **102 ms mean / 163 ms max**, first road
0.3 ms, second settlement 19 ms mean / 40 ms max, second road 0.3 ms.

Across more than **800 harness games** in this session: 0 errors, 0 timeouts, 0 crashes,
**0 games over budget**.

### 5. Risks

* **CPU.** The opening is the only expensive decision (≤163 ms here, i.e. ~75 ms on the scoring
  machine, against a 312 ms allowance at that point). Mid-game decisions are capped at 11 ms and
  the whole game averaged 350–460 ms here, i.e. ~1.0–1.3× F. The reserve (140 ms) and the 3 ms
  clock guard keep the clock safe; no game in ~500 went over. Watch the leaderboard's "games over
  the clock" column anyway.
* **F-specific opening.** `P3` (the default) assumes the opponent opens like F. Against a non-F
  opponent that assumption is wrong, which is exactly what `P4` (mix) and `P5` (generic) hedge.
  The two 20-game pool samples slightly favour P4, but the difference is inside the noise; if the
  group stage matters more than the F anchor, ship `P4` and re-measure.
* **Small-sample conclusions.** All preset differences are 40–80-game estimates (CI ±0.12–0.15).
  The only effects that are large enough to trust here are the opening (P1 0.650 vs P0 0.450) and
  "everything beats random".
* **Rules may not help.** P2 (rules without search) scored 0.500 vs P1's 0.650 in the 40-game
  screen, but the paired P3 vs P1 run favours P3 (+0.0875). Unresolved; the 400-game runs in
  section 3 settle it, and `v3_rules.py` is a single file that can be switched off with one flag.
* **The sweep writes `v3/weights_best.json`**, which the agent prefers over `weights.json`. That
  is intended, but remember it when comparing presets afterwards: delete the file to go back to
  the hand-set weights.

---

## Arena update 5 — version_6 (improving SUB3)

`catan-hackathon-agent/version_6/` is a **testing tree for SUB3** (`v5/sub3`: agent2 decides every
decision, and a friend-2 value-net ensemble overrides the main-turn pick when it is > 0.03 P(win)
better). Nothing here is a submission: no zip, no `make_submission`, and `my_team/`, `version_3/`,
`version4_nn/`, `v5/` and the kit are untouched.

### 1. Layout

| path | what it is |
|---|---|
| `version_6/base/` | a byte copy of `v5/sub3` — the baseline and the opponent for every screen |
| `version_6/v6/` | the working copy; every idea is one key in `SWITCHES` at the top of `agent.py` |
| `version_6/lab/` | the v5 tools, rewritten: `fastmatch.py` (clock-faithful match runner), `screen.py` (paired screen), `check_clock.py` (per-decision cost), `_load.py` (isolated agent imports), loaders, `opengen.py`/`train_open.py` (opening model), `gen_data.py`/`train_net.py` (value net) |
| `version_6/presets/<name>/` | one self-contained agent per step: `agent.py` at the top, `core.py` (a copy of `v6/agent.py`), all helpers and weights in the folder |
| `version_6/tools/sync_preset.py` | writes a preset folder from `v6/` plus its switch overrides |
| `version_6/EXPERIMENTS.md` | every experiment: hypothesis, setup, seeds, result, conclusion (failures kept) |

Every preset runs on its own:

```bash
python -m catan_kit.evaluate --a version_6/presets/s2_clock/agent.py --b examples/f_agent.py --games 1000
```

### 2. The lab had to be fixed before any number could be believed

Four defects, each found by a measurement:

1. **Helpers collided.** Two agents in one process both `import a2` / `import feats`, and `base/` and
   `v6/` each ship those files, so the second agent silently got the first agent's module. Fixed with
   `lab/_load.py` (import each agent with its own directory first on `sys.path`, pop the helper names
   from `sys.modules` afterwards). Check: `v6` with SUB3's switch values now makes **bit-identical
   decisions** to `base` — `diff +0.000 ±0.000` over 800 paired games.
2. **BLAS threads inflated the CPU measurement 3.5×.** The lab ran both agents in one process and
   charged `time.process_time()`, which also counts numpy's OpenBLAS worker threads spinning between
   calls: 97 ms/game pinned to one thread vs 345 ms/game unpinned, for the *same* agent. Fixed by
   pinning `OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=1` before numpy is imported (the harness pins
   OMP/MKL in `host.py`).
3. **`--cpu_scale` is the wrong clock model.** Scaling the charge while the agent times itself with
   `cpu_left` double-counts and invents over-budget games (11/800 at scale 2 for a change that is
   neutral at scale 1). Replaced by `--io_ms 2.5`, the per-decision JSON-view decode the harness
   charges outside `decide()` (measured: harness 299 ms/game vs 172 ms of in-decide CPU over 400
   games, 49 decisions each).
4. **`screen.py` printed candidate CPU 1000× too large.** Fixed.

After that the lab agrees with the harness on both win rate and CPU (table below), so the 800-game
paired screens are worth reading.

### 3. Step 1 — baseline: SUB3 under the harness clock

| run | games | win vs F | mean CPU | worst game | worst decision | decisions/game | net consulted | over budget |
|---|---|---|---|---|---|---|---|---|
| lab, first (broken) numbers | 800 | 0.790 ±0.028 | 451 ms | 1369 ms | 325 ms | 48 | 0.89 | 2 |
| **lab, fixed (`--clock`, `--io_ms 2.5`)** | 800 | **0.806 ±0.028** | **320 ms** | 690 ms | 31 ms | 45 | **1.00** | **0** |
| **harness, `catan_kit.evaluate`** | 400 | **0.7775 ±0.0408** | **299 ms** | 1020 ms | — | 49 | — | **0**, 0 errors |

**The "clock problem" was a measurement artifact.** With correct accounting SUB3 spends 320 ms of
the ~860 ms it is allowed at 45 decisions, never runs out, and already consults the net on 100% of
main turns. The lab is 7% above the harness on CPU — conservative, which is the right direction.

### 4. Step 2 — the net and the clock (preset `s2_clock`)

Profile first (`v6` under the harness clock, 10 games, idle sandbox, 47 decisions/game):

| part | ms per call | ms per game | share of `decide` |
|---|---|---|---|
| `net.scores` (sample world, 106 successor states, features, MLP) | 2.43 | 115 | 67% |
| … of which the MLP itself (`v4net.Value`, 106-row batch) | 0.20 | 9.4 | 5.5% |
| … of which `feats.features` | 0.59 | 28 | 16% |
| `a2._main` (copy + execute every action + planner) | 0.98 | 46.5 | 27% |
| `choose_opening` | 0.10 | 4.9 | 2.8% |
| **`decide` total** | **3.64** | **172** | 100% |

The net costs 2.4 ms per main turn against ~30 ms of allowance per decision. So the step became
"make the clock robust", not "make the net fit":

* `net_budget="adaptive"`: consult the net while `cpu_left > net_reserve + net_costs × (measured
  cost of one net call)` (~0.107 s) instead of SUB3's fixed `NET_MIN_CPU = 0.25 s`.
* `open_cap = 0.24`: the counter-opening search may spend at most `min(0.24, 0.6 × (cpu_left −
  0.04))` s. SUB3's search has no deadline: 5-30 ms typically, **233 ms measured** in one game, and
  2/800 games ran out of clock on the old, inflated numbers.
* `reserve` / `cap_frac` moved from module constants to instance settings (two agents with different
  settings can share a process; the harness is unaffected).

**Not done, and why (measurement, not laziness).** Sharing successor states between `a2._main` and
the net, and caching the value function's per-decision invariants, would save at most ~1.5 ms of a
3.6 ms decision — the overlap between the 46 ms `a2._main` and the 115 ms `net.scores`, of which
28 ms is features the net needs anyway. It would rewrite the two files that decide every game for
<10% of a budget the agent uses 37% of.

| 800 paired games vs `base` | win | diff ±CI | CPU | net | over budget |
|---|---|---|---|---|---|
| `base` (SUB3) | 0.806 | — | 320 ms | 1.00 | 0 |
| **`s2_clock`** | 0.805 | **−0.001 ±0.002** | 323 ms | 1.00 | 1 |
| `v6` @ SUB3 switches (control) | 0.806 | +0.000 ±0.000 | 324 ms | 1.00 | 0 |

`s2_clock` changes nothing about how the agent plays, within ±0.2 pt — but the opening can no longer
outrun the first allowance and the net threshold follows the real cost of a decision.

### 5. Step 3 — value-function weights (preset `s3_vf`)

Nine single-weight variants of `s2_clock`, 800 paired games each, seed0 500000, lab `--clock`:

| change (default → candidate) | win | diff ±CI | verdict |
|---|---|---|---|
| `hand_synergy` 100 → 50 | 0.809 | +0.004 ±0.015 | MAYBE |
| `hand_synergy` 100 → 25 / → 0 | 0.792 | −0.013 ±0.024 | DROP |
| `reachable_production_1` 10000 → 5000 | 0.806 | +0.001 ±0.016 | MAYBE |
| `reachable_production_1` 10000 → 2500 | 0.809 | +0.004 ±0.018 | MAYBE |
| `hand_resources` 1 → 0.5 / → 0 | 0.804 / 0.805 | −0.003 / −0.001 ±0.010 | DROP |
| `buildable_nodes` 1000 → 2000 | 0.812 | +0.006 ±0.013 | MAYBE |
| `buildable_nodes` 1000 → 5000 | 0.807 | +0.001 ±0.017 | MAYBE |

Nothing clears `diff − CI > 0`. The v5 numbers for the same weights (`SynLo +0.0675`,
`Reach1lo +0.0425`) do not reproduce on SUB3 + net: they were measured against a baseline where the
value function drove every decision. `hand_synergy` is *worse* when removed, so the term stays.
`s3_vf` ships `buildable_nodes 2000` (largest point estimate) with its CI in its README; it is not
recommended over `s2_clock` on strength.

### 6. Step 4 — margin and net breadth (preset `s4_net`)

| change | win | diff ±CI | verdict |
|---|---|---|---|
| `margin` 0.03 → 0.00 | 0.802 | −0.004 ±0.030 | DROP |
| `margin` 0.03 → 0.01 | 0.806 | +0.000 ±0.027 | DROP |
| net may override the robber rule | 0.771 | **−0.034 ±0.030** | DROP |
| net may override the pre-roll knight rule | 0.804 | −0.001 ±0.017 | DROP |
| net may override discards | 0.805 | +0.000 ±0.000 | no-op (engine has no `DISCARD_RESOURCE`) |
| agent2 planner budget ×2 (`cap_frac` 1/6 → 1/3) | 0.805 | +0.000 ±0.000 | no-op (deadline never binds) |

The net is a **worse robber placer** than agent2's hand-written denial rule — the largest single
loss in this campaign after the opening. Lowering the margin also widens the paired CI from 0.002 to
0.030 (the candidate diverges from the baseline, so the same 800 games resolve much less): margin
changes are expensive to test and showed nothing at this sample size.

### 7. Step 5 — win finder (preset `s5_win`)

`winfind_vp = 8`, depth 3, budget 400 game copies, gated on `cpu_left ≥ 0.12`:

| change | win | diff ±CI | CPU | over budget |
|---|---|---|---|---|
| run `winfind.find_win` from actual VP ≥ 8 | 0.807 | +0.003 ±0.003 | 356 ms (+38) | **3/800** |

DROP: the one-ply value function already puts `WIN` on any winning move and follows a 2-ply
continuation, so the explicit search adds nothing (same conclusion as v5's ±0), and it spends enough
CPU to push games over the limit. A final candidate must be at 0 over budget.

### 8. Step 6 — retrain the opening model

**Not run in the sandbox** (2 cores: generating 20k games would take ~2 h here; the 8-core command
is below). What *was* measured is the size of the prize:

| change | win | diff ±CI | verdict |
|---|---|---|---|
| learned counter-opening → plain 1-ply opening | 0.720 | **−0.085 ±0.037** | DROP |

The learned opening model is worth **+8.5 pt** on this agent — the largest effect in the campaign.
A retrain has a high bar (it must beat a model already worth 8.5 pt) and a real upside; it is the
first thing to run on the team machine. `lab/opengen.py` now takes `EXPLORER=<preset dir>` so the
mid-game explorer is the Step-2 agent instead of the v5 value function (verified with 40-game runs
in both modes).

### 9. Step 7 — value-net retrain (rejected)

`tools/gen_data.py`: 2000 games, seeds 100000+, self-play + vs F + vs base, 115,383 rows (2 seats ×
≤30 rows/game, evenly subsampled), 451 s. Split by **game seed** (`seed % 8 == 0`: 250 of 2000
games, 14,410 validation rows, win rate 0.519). `tools/train_net.py`: same architecture and file
format as the shipped nets (102→128→64→1, ReLU, one logit), 40 epochs, batch 512, Adam 3e-4, weight
decay 1e-4, 3 models.

| model | validation log-loss |
|---|---|
| **shipped ensemble** (mean of 3 logits) | **0.4505** |
| new models 0 / 1 / 2 | 0.4626 / 0.4648 / 0.4665 |
| new ensemble | 0.4596 (accuracy 0.7621 vs 0.7641) |

**REJECT**: 2000 games of self-play do not beat friend 2's weights; the acceptance rule (val
log-loss improves **and** the paired screen is KEEP) is not met, so `v6` keeps the original nets.
Also checked, and worth keeping: `feats.features` on the real game is bit-identical to
`feats.features` on the masked view at every decision (85 positions, max |diff| = 0) — the training
rows are exactly what the agent sees and the features leak no hidden information.

### 10. Final harness numbers, recommended preset, risks

Recommended: **`presets/s2_clock/`**. It is the only change that is both neutral in strength (bounded
at ±0.2 pt by 800 paired games) and removes two fragilities that only bite on a slower machine.
Nothing else in this campaign earned a KEEP.

Risks:
* **A plateau, not a gain.** No screened change beat SUB3 by more than the noise. The two data-hungry
  directions (opening model, value net) are prepared but not run: `s2_clock` is *not* stronger than
  SUB3, it is the same agent with a safer clock.
* `open_cap = 0.24` truncates the opening search in the (rare) games where it would take longer. On
  a much faster machine it never binds; on a much slower one it protects the first allowance.
* The net's cost is measured at run time (`net_costs × EWMA`); a pathological decision could push the
  threshold above `cpu_left` and skip the net near the end of a game. That is the intended trade.
* All screening is against `f_agent` on 500000+ seeds. A change that helps against F can hurt against
  a different opponent; the 400-game runs vs `version4_nn`, `f_k3` and `base` are in the table below.

### 11. Commands for the team machine (8 cores)

One per line, from your checkout root (`~/hyde_challenge`); `KIT=$PWD/catan-hackathon-agent`.
Every evaluation command runs from inside `catan-hackathon-agent/`.

```bash
cp -r ~/hyde_challenge/catan-hackathon-agent/version_6 ~/catan-agent-hackathon/version_6
cd ~/catan-agent-hackathon && python -m venv .venv && .venv/bin/pip install -r requirements.txt
cd ~/catan-agent-hackathon && nohup python -m catan_kit.evaluate --a version_6/presets/s2_clock/agent.py --b examples/f_agent.py --games 1000 --seed0 990000 > eval_s2clock_F.out 2>&1 &
cd ~/catan-agent-hackathon && nohup python -m catan_kit.evaluate --a version_6/presets/s2_clock/agent.py --b version4_nn/agent.py --games 400 --seed0 990000 > eval_s2clock_v4.out 2>&1 &
cd ~/catan-agent-hackathon && nohup python -m catan_kit.evaluate --a version_6/presets/s2_clock/agent.py --b version_3/opponents/f_k3/agent.py --games 400 --seed0 990000 > eval_s2clock_fk3.out 2>&1 &
cd ~/catan-agent-hackathon && nohup python -m catan_kit.evaluate --a version_6/presets/s2_clock/agent.py --b version_6/base/agent.py --games 400 --seed0 990000 > eval_s2clock_base.out 2>&1 &
cd ~/catan-agent-hackathon/version_6/lab && KIT=$PWD/../.. PYTHONHASHSEED=0 AGENT_PATH=../presets/s2_clock/agent.py nohup python fastmatch.py anyagent:Agent f_agent:Agent 800 990000 8 --clock > match_s2clock.out 2>&1 &
cd ~/catan-agent-hackathon/version_6/lab && KIT=$PWD/../.. PYTHONHASHSEED=0 SWV6_LIST='[{"vf":{"buildable_nodes":2000}},{"vf":{"hand_synergy":50}}]' nohup python screen.py 400 600000 8 baseagent:Agent swagent:A0 swagent:A1 --clock > screen_fresh.out 2>&1 &
cd ~/catan-agent-hackathon/version_6/lab && KIT=$PWD/../.. PYTHONHASHSEED=0 EXPLORER=../presets/s2_clock nohup python opengen.py open_20k_v6.npz 20000 100000 8 > opengen_v6.out 2>&1 &
cd ~/catan-agent-hackathon/version_6/lab && KIT=$PWD/../.. PYTHONHASHSEED=0 nohup python train_open.py open_20k_v6.npz > train_open_v6.out 2>&1 &
cd ~/catan-agent-hackathon && nohup python version_6/tools/gen_data.py --games 8000 --workers 8 --chunk 50 --out version_6/data8k > gen_data_8k.out 2>&1 &
cd ~/catan-agent-hackathon && nohup python version_6/tools/train_net.py --data "version_6/data8k/shard_*.npz" --out version_6/nets8k --val-mod 8 --epochs 80 --models 3 > train_net_8k.out 2>&1 &
tail -5 eval_s2clock_F.out && tail -3 train_net_8k.out
```

Notes for those runs: `screen.py` compares every `swagent:A<i>` against one baseline run on the same
boards; `SWV6_LIST` is the JSON list of switch sets (see `EXPERIMENTS.md` for the sets used), and
`AGENT_PATH` lets `anyagent:Agent` load any preset file. A retrained opening model goes into a preset with
`cp open_logreg.json ../presets/<name>/open_logreg.json`, a retrained value net with
`cp nets8k/value_w*.npz ../presets/<name>/`; keep a preset only if its held-out numbers improve
**and** `screen.py` says KEEP.

### 12. Final harness runs of `s2_clock` (seeds 990000+)

| opponent | games | win rate | mean CPU/game | worst game | over budget | errors |
|---|---|---|---|---|---|---|
| `examples/f_agent.py` | 1000 | **0.7900 ±0.0252** | 283.7 ms | 940 ms | **0** | **0** (48,160 decisions) |
| `version_6/base/agent.py` (SUB3) | 400 | **0.5025 ±0.0490** | 278.9 ms | 810 ms | **0** | **0** (19,327 decisions) |
| `version4_nn/agent.py` | 400 | **0.6000 ±0.0480** | 319.7 ms | 1020 ms | **0** | **0** (20,574 decisions) |
| `version_3/opponents/f_k3/agent.py` | 400 | **0.8125 ±0.0383** | 279.4 ms | 900 ms | **0** | **0** (19,092 decisions) |

Zero over-budget games, zero errors, zero timeouts/crashes, 0.04 s load time, worst game 1020 ms
against a limit of 1500 ms.  Against F, 0.7900 ±0.0252 is SUB3's own level (SUB3: 0.794 ±0.028 over
800 games on the team machine, 0.7775 ±0.0408 over 400 in this sandbox) — consistent with the paired
lab screen's −0.1 pt.  The head-to-head against the baseline settles it: **50.25% ±4.9 over 400
games**, i.e. the same agent.  0.600 vs `version4_nn` is a fresh-seed number (SUB3 measured 0.642 on
its own seeds; the two CIs overlap) and 0.8125 vs `f_k3`.

Run in this sandbox with `python -m catan_kit.evaluate` (one process, `--seed0 990000`); the team
machine is ~1.3× faster, so expect smaller CPU numbers there.
