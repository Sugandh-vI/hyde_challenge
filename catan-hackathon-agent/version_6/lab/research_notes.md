# Research notes (literature + catanatron source), 17 Oct 11:40

Tags: [S] a source states it, [C] read in the catanatron source at d3f4ad05, [I] inference.

## What we use
1. **Expectimax search, not max over chance.** catanatron `AlphaBetaPlayer` averages chance outcomes by probability (11 dice sums, unseen dev deck, 1/5 steal) and treats the opponent's turn as a min node [C]. The docs leaderboard reports AlphaBeta(depth=2) beats ValueFunction 80%, but on only 25 games [S, weak]. Our naive same-turn max-over-children depth 2 lost (46%), consistent with the winner's curse when maxing over chance.
   Source: docs.catanatron.com/advanced/making-catanatron-stronger.md, `players/minimax.py`, `tree_search_utils.py`.
2. **F's blind spots, confirmed in `value.py`** [C]: no port term; Longest Road worth 0.1×length while any buildable node exists; no opponent-VP term; discard penalty −5 (negligible); greedy on any VP (3e14); chance moves judged on one random sample. This matches our measurement that the LR holder wins 87% of F-vs-F games.
3. **Spend CPU unevenly** [I]: forced moves are free. Spend on setup, the robber, and build/trade/end-turn choices.
4. **Linear afterstate value fitted to outcomes** (Buro 2002; TD-Gammon) [S]: the same recipe as our opening logistic, extended to mid-game positions. Overnight, a greedy learned value lost in our own sim, so gate it.
5. **JSettlers "win-game ETA"** [S]: turns to the next settlement/city/LR from production plus port rates. A candidate feature.
6. **Overfitting risk** [S]: genetic-programming bots tuned against one JSettlers bot did poorly against another (Carleton "Evolvers of Catan", 2021). Validate against a pool.
7. **How O was probably built** [I, from C]: catanatron's `VRLPlayer` is 1-ply over a value model on `create_sample_vector`. That vector **lacks** production, enemy production, reachability, buildable nodes, tiles touched and hand synergy. Likely still 1-ply and port/LR-blind.

## Do not
- MCTS or rollouts at our budget: on the catanatron leaderboard, ValueFunction beats GreedyPlayouts(25) 90% [S]. Cazenave's Deep Catan: 10k-simulation MCTS is still beaten without heuristics [S].
- Max over chance outcomes.
- Retune only F's 13 weights at small ranges (Optuna 0-100 and SPSA in the repo; no recorded gains) [C]. Add features instead.
- Dev / Largest Army focus (our measurement: 44%).
- Deep RL from scratch in one day (Gendre & Kaneko 2020; Cazenave ExIt needs ~50k states per iteration).

## Sources
github.com/bcollazo/catanatron @ d3f4ad05 (value.py, minimax.py, tree_search_utils.py, features.py, catanatron_experimental spsa.py / optunation.py / reinforcement.py); docs.catanatron.com; lamsade.dauphine.fr/~cazenave/papers/DeepCatan.pdf; Carleton CS comps 2021 "Evolvers of Catan"; arXiv:2008.07079 (Gendre & Kaneko).
