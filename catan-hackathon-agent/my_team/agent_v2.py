"""Agent v2: F's value function + same-turn search with a CPU budget.

Design (see my_team/README notes and EXPERIMENTS.md):
  * value function: catanatron base_fn with weights from weights.json next to this file (falls
    back to DEFAULT_WEIGHTS), plus our own hidden victory-point cards;
  * each decision samples K worlds with obs.sample_world(self.rng) (K = 1 by default);
  * candidate actions are pruned with catanatron's list_prunned_actions;
  * iterative deepening over OUR OWN actions inside the current turn: depth 1 is exactly F's
    one-step evaluation, deeper levels use a beam and stop the line at END_TURN / chance events;
  * the per-decision budget comes from obs.cpu_left, with a reserve for the rest of the game.

Every failure path returns a legal index; decide() never raises.
"""

import os
import random
import time

# ---- Knobs (constants at the top of the file) ----
K = 1                 # worlds sampled per decision
BEAM_WIDTH = 3        # children kept per node when deepening
ROOT_BEAM = 6         # root actions that get deepened
MAX_DEPTH = 3         # maximum number of our own actions in a line
RESERVE_S = 0.12      # CPU kept for the rest of the game
CAP_NORMAL_S = 0.030  # per-decision cap outside the opening
CAP_OPENING_S = 0.055 # per-decision cap during the initial placement
OPENING_TURNS = 4

try:
    from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn
except Exception:                                            # pragma: no cover
    DEFAULT_WEIGHTS, get_value_fn = {}, None

try:
    from search import (                                     # helper next to this file
        load_weights, make_value_fn, compute_budget, choose_action,
        BEAM_WIDTH as _BEAM, ROOT_BEAM as _ROOT, MAX_DEPTH as _DEPTH,
        RESERVE_S as _RES, CAP_NORMAL_S as _CAP_N, CAP_OPENING_S as _CAP_O,
        K_WORLDS as _K,
    )
    BEAM_WIDTH, ROOT_BEAM, MAX_DEPTH = _BEAM, _ROOT, _DEPTH
    RESERVE_S, CAP_NORMAL_S, CAP_OPENING_S, K = _RES, _CAP_N, _CAP_O, _K
    _HAS_SEARCH = True
except Exception:                                            # pragma: no cover
    _HAS_SEARCH = False

    def load_weights(folder=None):
        try:
            return dict(DEFAULT_WEIGHTS)
        except Exception:
            return {}

    def make_value_fn(weights=None):
        try:
            return get_value_fn("base_fn", weights or DEFAULT_WEIGHTS)
        except Exception:
            return None

    def compute_budget(cpu_left, is_opening):
        try:
            cpu_left = float(cpu_left) if cpu_left is not None else 1.0
        except Exception:
            cpu_left = 1.0
        if is_opening:
            return min(CAP_OPENING_S, max(0.010, cpu_left * 0.16))
        avail = max(0.0, cpu_left - RESERVE_S)
        return max(0.0, min(avail / 30.0, CAP_NORMAL_S, cpu_left * 0.75))

    def choose_action(worlds, me, playable_actions, value_fn, budget_s, **_kw):
        world = worlds[0]
        best_i, best_v = 0, None
        for i, action in enumerate(playable_actions):
            try:
                succ = world.copy()
                succ.execute(action)
                v = value_fn(succ, me)
                if best_v is None or v > best_v:
                    best_i, best_v = i, v
            except Exception:
                continue
        return best_i, (best_v if best_v is not None else 0.0)


class Agent:
    def __init__(self):
        self.rng = random.Random(0)
        self.value_fn = None
        self.weights = {}
        try:
            self.weights = load_weights(os.path.dirname(os.path.abspath(__file__)))
            self.value_fn = make_value_fn(self.weights)
        except Exception:
            self.value_fn = None
        if self.value_fn is None:
            try:
                self.value_fn = get_value_fn("base_fn", DEFAULT_WEIGHTS)
            except Exception:
                self.value_fn = None

    # ------------------------------------------------------------------
    def _fallback(self, obs, playable_actions):
        """Cheap F one-step on one sampled world. Used only when the main path fails."""
        try:
            world = obs.sample_world(self.rng)
        except Exception:
            return 0
        best_i, best_v = 0, None
        for i, action in enumerate(playable_actions):
            try:
                succ = world.copy()
                succ.execute(action)
                if succ.winning_color() is not None and succ.winning_color() == obs.me:
                    return i
                v = self.value_fn(succ, obs.me)
                if best_v is None or v > best_v:
                    best_i, best_v = i, v
            except Exception:
                continue
        return best_i

    # ------------------------------------------------------------------
    def decide(self, obs, playable_actions) -> int:
        try:
            n = len(playable_actions)
            if n == 0:
                return 0
            if n == 1:
                return 0
            if self.value_fn is None:
                return 0

            # Clock safety net: with only a few ms left, a full depth-1 pass (~10 ms) could
            # overspend the allowance and make the harness play random moves for the rest of the
            # game.  End the turn if that is legal, otherwise take the first legal action.
            cpu_left = getattr(obs, "cpu_left", None)
            if cpu_left is not None:
                try:
                    if float(cpu_left) < 0.003:
                        for i, a in enumerate(playable_actions):
                            if getattr(a.action_type, "name", "") == "END_TURN":
                                return i
                        return 0
                except Exception:
                    pass

            # Sample the worlds first: is_opening and all search happen on them.
            k = max(1, int(K))
            worlds = []
            for _ in range(k):
                try:
                    worlds.append(obs.sample_world(self.rng))
                except Exception:
                    break
            if not worlds:
                return 0
            world = worlds[0]

            is_opening = False
            try:
                is_opening = bool(world.state.is_initial_build_phase)
            except Exception:
                pass
            try:
                turn = getattr(obs, "turn", None)
                if turn is not None and turn <= OPENING_TURNS:
                    is_opening = True
            except Exception:
                pass

            budget = compute_budget(getattr(obs, "cpu_left", None), is_opening)
            if budget <= 0.0:
                return self._fallback(obs, playable_actions)

            idx, _val = choose_action(
                worlds, obs.me, playable_actions, self.value_fn, budget,
                beam=BEAM_WIDTH, root_beam=ROOT_BEAM, max_depth=MAX_DEPTH,
            )
            if type(idx) is int and 0 <= idx < n:
                return idx
            return self._fallback(obs, playable_actions)
        except BaseException:
            # decide() must never raise and must never end the game on a random move.
            try:
                return self._fallback(obs, playable_actions)
            except BaseException:
                return 0
