"""Combination of friend 2 (v4 value-net expectimax) and friend 3 (agent2 rules)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sub3"))
import a2 as agent2
from v4agent import Agent as V4

AT = agent2.AT


class Combo(V4):
    """v4 for the main turn; agent2 for setup, robber placement and the pre-roll knight decision."""
    def __init__(self):
        super().__init__()
        self.a2 = agent2.Agent()

    def decide(self, obs, actions):
        try:
            types = {a.action_type for a in actions}
            st = obs._game.state
            me = obs.me
            opp = next(c for c in st.colors if c != me)
            if st.is_initial_build_phase or AT.MOVE_ROBBER in types or AT.ROLL in types:
                return self.a2.decide(obs, actions)
        except Exception:
            pass
        return super().decide(obs, actions)


class ComboSetupOnly(V4):
    """v4 everywhere except setup (agent2's counter-opening + roads)."""
    def __init__(self):
        super().__init__()
        self.a2 = agent2.Agent()

    def decide(self, obs, actions):
        if obs._game.state.is_initial_build_phase:
            return self.a2.decide(obs, actions)
        return super().decide(obs, actions)


class A2NetMain(agent2.Agent):
    """agent2 everywhere, but its main-turn choice is replaced by v4 when v4 is confident:
    v4's best action beats agent2's pick by > 0.03 in estimated P(win)."""
    MARGIN = 0.03

    def __init__(self):
        super().__init__()
        self.v4 = V4()

    def _main(self, world, me, actions, deadline):
        i = super()._main(world, me, actions, deadline)
        try:
            vals = self.v4.scores(self._obs, actions, world=world)
            j = int(vals.argmax())
            if j != i and vals[j] - vals[i] > self.MARGIN:
                return j
        except Exception:
            pass
        return i

    def decide(self, obs, actions):
        self._obs = obs
        return super().decide(obs, actions)
