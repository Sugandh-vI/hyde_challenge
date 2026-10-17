"""Experiment E1: F everywhere, except a balance/port-aware opening (settlements + setup roads)."""
import random

from catanatron.models.enums import ActionType, SETTLEMENT, CITY
from catanatron.players.value import DEFAULT_WEIGHTS, base_fn

VALUE_FN = base_fn(DEFAULT_WEIGHTS)
RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
W = dict(new_res=3.0, pair=1.0, port2=1.5, port3=1.0)


def my_prod(board, nodes):
    tot = {r: 0.0 for r in RES}
    for n in nodes:
        for r, p in board.map.node_production[n].items():
            tot[r] += p * 36
    return tot


def spot_score(board, node, mine):
    have = my_prod(board, mine)
    add = {r: p * 36 for r, p in board.map.node_production[node].items()}
    after = {r: have[r] + add.get(r, 0.0) for r in RES}
    s = sum(add.values())
    s += W["new_res"] * sum(1 for r in add if have[r] == 0 and add[r] > 0)
    s += W["pair"] * (min(after["WOOD"], after["BRICK"]) - min(have["WOOD"], have["BRICK"]))
    s += W["pair"] * (min(after["WHEAT"], after["ORE"]) - min(have["WHEAT"], have["ORE"]))
    for r, nodes in board.map.port_nodes.items():
        if node in nodes:
            s += W["port3"] if r is None else W["port2"] * after.get(r, 0.0) / 5.0
    return s


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, actions):
        st = obs._game.state
        board = obs.board
        a0 = actions[0].action_type
        if st.is_initial_build_phase and a0 == ActionType.BUILD_SETTLEMENT:
            b = st.buildings_by_color[obs.me]
            mine = b[SETTLEMENT] + b[CITY]
            return max(range(len(actions)), key=lambda i: spot_score(board, actions[i].value, mine))
        if st.is_initial_build_phase and a0 == ActionType.BUILD_ROAD:
            b = st.buildings_by_color[obs.me]
            mine = b[SETTLEMENT] + b[CITY]
            open_nodes = set(board.buildable_node_ids(obs.me, initial_build_phase=True))

            def road_score(i):
                u, v = actions[i].value
                far = v if u in mine else u
                cands = [n for n in _neighbors(board, far) if n in open_nodes]
                return max((spot_score(board, n, mine) for n in cands), default=0.0)
            return max(range(len(actions)), key=road_score)
        world = obs.sample_world(self.rng)
        best_i, best_v = 0, None
        for i, a in enumerate(actions):
            g = world.copy()
            try:
                g.execute(a)
            except Exception:
                continue
            v = VALUE_FN(g, obs.me)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i


def _neighbors(board, node):
    import networkx  # noqa: F401
    from catanatron.models.board import STATIC_GRAPH
    return list(STATIC_GRAPH.neighbors(node))
