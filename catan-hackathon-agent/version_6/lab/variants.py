"""1-ply value agents with different weights, for hypothesis tests vs F. F's own opening is kept
(opening is tested separately), so each variant isolates one mid-game idea."""
import random

from vf import make_vf


class _VAgent:
    W = {}

    def __init__(self):
        self.rng = random.Random(0)
        self.vf = make_vf(self.W)

    def decide(self, obs, actions):
        world = obs.sample_world(self.rng)
        best_i, best_v = 0, None
        for i, a in enumerate(actions):
            g = world.copy()
            try:
                g.execute(a)
            except Exception:
                continue
            v = self.vf(g, obs.me)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i


class Same(_VAgent):          # sanity: must equal F (expect 50.0% exactly-ish)
    W = {}


class DevLo(_VAgent):         # value dev cards / knights above roads' buildable-node term
    W = dict(dev_bonus=2e3, army_bonus=5e3)


class DevHi(_VAgent):         # ... above roads' reachable-production term too
    W = dict(dev_bonus=2e4, army_bonus=5e4)


class RoadLo(_VAgent):
    W = dict(lr_always=2e3)


class RoadHi(_VAgent):
    W = dict(lr_always=2e4)


class OppVP(_VAgent):         # care about the opponent's VPs (deny LR/LA swings)
    W = dict(opp_vp=-1e14)


class Ports(_VAgent):
    W = dict(port2=3e7, port3=1e6)


class Road5(_VAgent):
    W = dict(lr_always=5e2)


class RoadPorts(_VAgent):
    W = dict(lr_always=2e3, port2=3e7, port3=1e6)


class RoadPortsOpen(RoadPorts):
    """RoadPorts mid-game + the balance/port-aware opening of open_agent."""
    def decide(self, obs, actions):
        st = obs._game.state
        if st.is_initial_build_phase:
            from open_agent import Agent as OA
            if not hasattr(self, "_oa"):
                self._oa = OA()
            return self._oa.decide(obs, actions)
        return super().decide(obs, actions)


class CounterOpen(RoadPorts):
    """RoadPorts mid-game + counter-opening (search vs predicted F replies); roads from open_agent."""
    def decide(self, obs, actions):
        st = obs._game.state
        if st.is_initial_build_phase:
            from catanatron.models.enums import ActionType, SETTLEMENT
            if actions[0].action_type == ActionType.BUILD_SETTLEMENT:
                import opening
                b = st.buildings_by_color
                opp = next(c for c in st.colors if c != obs.me)
                mine, theirs = b[obs.me][SETTLEMENT], b[opp][SETTLEMENT]
                i_first = st.colors[0] == obs.me
                node = opening.choose(obs.board, [a.value for a in actions], mine, theirs, i_first,
                                      k1=getattr(self, 'K1', 6), k2=getattr(self, 'K2', 10))
                return next(i for i, a in enumerate(actions) if a.value == node)
            from open_agent import Agent as OA
            if not hasattr(self, "_oa"):
                self._oa = OA()
            return self._oa.decide(obs, actions)
        return super().decide(obs, actions)


def robber_score(board, coord, me, opp, opp_hand, self_pen=4.0, need_bonus=3.0):
    from catanatron.models.map import number_probability
    tile = board.map.land_tiles[coord]
    if tile.resource is None:
        return -1e9
    pips = number_probability(tile.number) * 36
    s = 0.0
    for n in tile.nodes.values():
        b = board.buildings.get(n)
        if b is None:
            continue
        lvl = 2 if b[1] == "CITY" else 1
        s += pips * lvl * (1 if b[0] == opp else -self_pen)
    if s > 0 and opp_hand.get(tile.resource, 0) == 0:
        s += need_bonus
    return s


class CounterRobber(CounterOpen):
    """CounterOpen + explicit robber placement (hex choice); victim = opponent when possible."""
    def decide(self, obs, actions):
        from catanatron.models.enums import ActionType
        if actions[0].action_type == ActionType.MOVE_ROBBER:
            opp = next(c for c in obs._game.state.colors if c != obs.me)
            def key(i):
                coord, victim = actions[i].value
                return (robber_score(obs.board, coord, obs.me, opp, obs.opp_resources), victim == opp)
            return max(range(len(actions)), key=key)
        return super().decide(obs, actions)


class _AB:
    """catanatron AlphaBetaPlayer (expectimax over chance, min for opponent) with our leaf vf."""
    depth = 2
    W = dict(lr_always=2e3, port2=3e7, port3=1e6)

    def _ab(self, color):
        if getattr(self, "_abp", None) is None or self._abp.color != color:
            from catanatron.players.minimax import AlphaBetaPlayer
            p = AlphaBetaPlayer(color, depth=self.depth, prunning=True)
            vf = make_vf(self.W)
            p.use_value_function = True
            p.value_function = vf
            self._abp = p
        return self._abp


class ABCounter(CounterOpen, _AB):
    def decide(self, obs, actions):
        if obs._game.state.is_initial_build_phase:
            return super().decide(obs, actions)
        world = obs.sample_world(self.rng)
        a = self._ab(obs.me).decide(world, actions)
        return next((i for i, b in enumerate(actions) if b == a), 0)


class ABCounterF(ABCounter):
    W = {}          # F's exact leaf, to isolate the search effect


class CounterRoadV(CounterOpen):
    """Counter-opening settlements, but setup roads chosen by our 1-ply value function."""
    def decide(self, obs, actions):
        from catanatron.models.enums import ActionType
        st = obs._game.state
        if st.is_initial_build_phase and actions[0].action_type == ActionType.BUILD_ROAD:
            return _VAgent.decide(self, obs, actions)
        return super().decide(obs, actions)


class CounterRoadReach(CounterOpen):
    """Setup road: toward the best open spot, plus room to keep extending (LR potential)."""
    def decide(self, obs, actions):
        from catanatron.models.enums import ActionType, SETTLEMENT, CITY
        from catanatron.models.board import STATIC_GRAPH
        from open_agent import spot_score
        st = obs._game.state
        if st.is_initial_build_phase and actions[0].action_type == ActionType.BUILD_ROAD:
            board = st.board
            b = st.buildings_by_color[obs.me]
            mine = b[SETTLEMENT] + b[CITY]
            open_nodes = set(board.buildable_node_ids(obs.me, initial_build_phase=True))
            occupied = set(board.buildings)

            def score(i):
                u, v = actions[i].value
                far = v if u in mine else u
                near = [n for n in STATIC_GRAPH.neighbors(far) if n in open_nodes]
                best = max((spot_score(board, n, mine) for n in near), default=0.0)
                ring2 = {m for n in STATIC_GRAPH.neighbors(far) for m in STATIC_GRAPH.neighbors(n)}
                room = sum(1 for n in ring2 if n in open_nodes and n not in occupied)
                return best + 0.5 * room
            return max(range(len(actions)), key=score)
        return super().decide(obs, actions)


class CounterWide(CounterOpen):
    K1, K2 = 12, 20


class CounterNarrow(CounterOpen):
    K1, K2 = 3, 6


class CounterWin(CounterOpen):
    """CounterOpen + win finder when actual VP >= 8."""
    def decide(self, obs, actions):
        st = obs._game.state
        if not st.is_initial_build_phase and obs.my_actual_vps >= 8:
            from winfind import find_win
            world = obs.sample_world(self.rng)
            a = find_win(world, obs.me)
            if a is not None:
                for i, b in enumerate(actions):
                    if b == a:
                        return i
        return super().decide(obs, actions)
