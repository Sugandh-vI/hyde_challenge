"""How often does fmodel.f_pick match real F's setup settlement? Plays F-vs-F setups only."""
import os
import random
import sys

sys.path.insert(0, os.environ.get("KIT", ""))
import catan_kit  # noqa: F401
from catanatron.game import Game
from catanatron.models.enums import ActionType, SETTLEMENT
from catanatron.models.player import Color, Player
from catanatron.players.value import DEFAULT_WEIGHTS, base_fn
from fmodel import f_pick

VF = base_fn(DEFAULT_WEIGHTS)
hits = [0, 0]
gaps = []


class F(Player):
    def decide(self, game, actions):
        best, bv = actions[0], None
        for a in actions:
            g = game.copy()
            g.execute(a)
            v = VF(g, self.color)
            if bv is None or v > bv:
                best, bv = a, v
        if game.state.is_initial_build_phase and actions[0].action_type == ActionType.BUILD_SETTLEMENT:
            st = game.state
            mine = st.buildings_by_color[self.color][SETTLEMENT]
            occ = [n for c in st.colors for n in st.buildings_by_color[c][SETTLEMENT]]
            pred = f_pick(st.board, mine, occ, [a.value for a in actions])
            hits[0] += pred == best.value
            if pred != best.value:
                from fmodel import prod_value
                ap = next(a for a in actions if a.value == pred)
                g1 = game.copy(); g1.execute(ap); g2 = game.copy(); g2.execute(best)
                gaps.append("%.3g" % (VF(g2, self.color) - VF(g1, self.color)))
            hits[1] += 1
        return best


for s in range(300):
    random.seed(s)
    g = Game(players=[F(Color.RED), F(Color.BLUE)], seed=800000 + s)
    while g.state.is_initial_build_phase:
        g.play_tick()
print(f"f_pick matches F on {hits[0]}/{hits[1]} setup settlements ({hits[0]/hits[1]:.3f})")
import collections
print(collections.Counter(gaps).most_common(8))
