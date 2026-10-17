"""Opponent for the pool: F with a 10% chance of a uniformly random legal move.

Robustness check: an agent that overfits to F's deterministic play should lose ground against an
opponent that deviates from F one time in ten.
"""
import random

from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn

VALUE_FN = get_value_fn("base_fn", DEFAULT_WEIGHTS)
EPS = 0.10


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        try:
            n = len(playable_actions)
            if n <= 1:
                return 0
            if self.rng.random() < EPS:
                return self.rng.randrange(n)
            world = obs.sample_world(self.rng)
            best_i, best_v = 0, None
            for i, action in enumerate(playable_actions):
                try:
                    succ = world.copy()
                    succ.execute(action)
                    v = VALUE_FN(succ, obs.me)
                except Exception:
                    continue
                if best_v is None or v > best_v:
                    best_i, best_v = i, v
            return best_i
        except Exception:
            return 0
