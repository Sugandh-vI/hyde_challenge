"""Opponent for the pool: F's value function averaged over 3 sampled worlds.

Same 1-ply rule as examples/f_agent.py, but the score of an action is the mean value over three
worlds drawn with obs.sample_world, instead of one.  This is the "sample several worlds and
average" advice from the problem statement, and it makes the opponent stronger against hidden
information without any other change.
"""
import random

from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn

VALUE_FN = get_value_fn("base_fn", DEFAULT_WEIGHTS)


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        try:
            if len(playable_actions) <= 1:
                return 0
            worlds = []
            for _ in range(3):
                try:
                    worlds.append(obs.sample_world(self.rng))
                except Exception:
                    break
            if not worlds:
                return 0
            best_i, best_v = 0, None
            for i, action in enumerate(playable_actions):
                total, n = 0.0, 0
                for world in worlds:
                    try:
                        succ = world.copy()
                        succ.execute(action)
                        total += VALUE_FN(succ, obs.me)
                        n += 1
                    except Exception:
                        continue
                if n == 0:
                    continue
                v = total / n
                if best_v is None or v > best_v:
                    best_i, best_v = i, v
            return best_i
        except Exception:
            return 0
