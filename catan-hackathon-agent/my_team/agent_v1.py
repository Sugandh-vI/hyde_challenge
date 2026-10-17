"""Agent v1: safe copy of F with win-check and try/except fallback."""

import os
import json
import random

try:
    from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn
except Exception:
    DEFAULT_WEIGHTS = {}
    get_value_fn = None

# Try to load weights.json if present, else use DEFAULT_WEIGHTS
def _load_value_fn():
    weights = DEFAULT_WEIGHTS
    try:
        p = os.path.join(os.path.dirname(__file__), "weights.json")
        if os.path.exists(p):
            with open(p, "r") as f:
                w = json.load(f)
                # validate keys
                if isinstance(w, dict) and len(w) >= 5:
                    weights = w
    except Exception:
        weights = DEFAULT_WEIGHTS
    try:
        if get_value_fn is not None:
            return get_value_fn("base_fn", weights)
    except Exception:
        pass
    return None

VALUE_FN = _load_value_fn()
DEFAULT_FALLBACK_WEIGHTS = DEFAULT_WEIGHTS


class Agent:
    def __init__(self):
        self.rng = random.Random(0)
        # cache value fn in instance to allow per-instance loading (e.g., tuned weights)
        self.value_fn = VALUE_FN
        if self.value_fn is None:
            try:
                from catanatron.players.value import DEFAULT_WEIGHTS as DW, get_value_fn as GV
                self.value_fn = GV("base_fn", DW)
            except Exception:
                self.value_fn = None

    def decide(self, obs, playable_actions) -> int:
        # Hard fallback: if anything goes wrong, return index 0
        try:
            if not playable_actions:
                return 0
            if len(playable_actions) == 1:
                return 0

            # Winning move check: if any action leads to actual VPs >=10, play it immediately
            # We need to sample a world to test. Use try/except per action so errors don't crash.
            try:
                world = obs.sample_world(self.rng)
                # Determine our player key for VP check
                # Use obs.me color; after execute we can check winning_color or actual VPs
                for idx, action in enumerate(playable_actions):
                    try:
                        succ = world.copy()
                        succ.execute(action)
                        # Check win via engine's winning_color or actual VPs
                        if succ.winning_color() is not None and succ.winning_color() == obs.me:
                            return idx
                        # Also check actual VPs directly if available
                        try:
                            from catanatron.state_functions import player_key
                            key = player_key(succ.state, obs.me)
                            actual = succ.state.player_state.get(f"{key}_ACTUAL_VICTORY_POINTS", None)
                            if actual is not None and actual >= 10:
                                return idx
                            # Also check public + hidden? The engine tracks actual, so above is enough.
                            # Fallback: check if any player's actual >=10 and winner is us
                            # Some versions store victory points differently; check both
                            vp_public = succ.state.player_state.get(f"{key}_VICTORY_POINTS", 0)
                            # If vp_public >=10 and winning_color is us, already handled.
                        except Exception:
                            pass
                    except Exception:
                        continue
            except Exception:
                # sample_world failed, fallback to F logic without win check
                pass

            # Standard F logic: one-step value evaluation
            try:
                world = obs.sample_world(self.rng)
            except Exception:
                # If sampling fails, just return 0
                return 0

            best_index = 0
            best_value = None
            # VALUE_FN is already bound; but use instance's copy if available
            vf = self.value_fn if self.value_fn is not None else VALUE_FN
            if vf is None:
                return 0

            for index, action in enumerate(playable_actions):
                try:
                    successor = world.copy()
                    successor.execute(action)
                except Exception:
                    continue
                try:
                    value = vf(successor, obs.me)
                except Exception:
                    continue
                if best_value is None or value > best_value:
                    best_index, best_value = index, value

            # best_value may stay None if all executes failed, then fallback 0
            return best_index

        except Exception:
            # Global fallback: never raise
            try:
                return 0
            except Exception:
                return 0
