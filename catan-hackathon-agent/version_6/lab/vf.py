"""Parametrised value function: catanatron's base_fn (F) plus extra terms F lacks.

Every extra weight defaults to 0 and `lr_mode` defaults to F's rule, so VF(dict()) == F exactly.
Extra terms:
  lr_always   longest-road length, always (F uses 0.1 unless no buildable node is left)
  opp_vp      opponent public VPs (F ignores them)
  port2/port3 owned 2:1 port weighted by my production of that resource / owned 3:1 port
  dev_bonus   extra per dev card in hand; army_bonus extra per knight played
"""
from catanatron.features import build_production_features, reachability_features, resource_hand_features
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY
from catanatron.players.value import DEFAULT_WEIGHTS, value_production
from catanatron.state_functions import (get_longest_road_length, get_played_dev_cards, player_key,
                                        player_num_dev_cards, player_num_resource_cards)

EXTRA = dict(lr_always=0.0, opp_vp=0.0, port2=0.0, port3=0.0, dev_bonus=0.0, army_bonus=0.0)
_PROD = build_production_features(True)


def make_vf(overrides=None):
    p = dict(DEFAULT_WEIGHTS)
    p.update(EXTRA)
    p.update(overrides or {})

    def fn(game, color):
        st = game.state
        sample = _PROD(game, color)
        production = value_production(sample, "P0")
        enemy_production = value_production(sample, "P1", False)
        key = player_key(st, color)
        lr_len = get_longest_road_length(st, color)
        reach = reachability_features(game, color, 2)
        r0 = sum(reach[f"P0_0_ROAD_REACHABLE_{r}"] for r in RESOURCES)
        r1 = sum(reach[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES)
        hand = resource_hand_features(game, color)
        d_city = (max(2 - hand["P0_WHEAT_IN_HAND"], 0) + max(3 - hand["P0_ORE_IN_HAND"], 0)) / 5.0
        d_sett = sum(max(1 - hand[f"P0_{r}_IN_HAND"], 0) for r in ("WHEAT", "SHEEP", "BRICK", "WOOD")) / 4.0
        synergy = (2 - d_city - d_sett) / 2
        n_hand = player_num_resource_cards(st, color)
        owned = st.buildings_by_color[color][SETTLEMENT] + st.buildings_by_color[color][CITY]
        tiles = set()
        for n in owned:
            tiles.update(st.board.map.adjacent_tiles[n])
        n_build = len(st.board.buildable_node_ids(color))
        lr_factor = p["longest_road"] if n_build == 0 else 0.1
        n_dev = player_num_dev_cards(st, color)
        knights = get_played_dev_cards(st, color, "KNIGHT")
        v = (st.player_state[f"{key}_VICTORY_POINTS"] * p["public_vps"]
             + production * p["production"] + enemy_production * p["enemy_production"]
             + r0 * p["reachable_production_0"] + r1 * p["reachable_production_1"]
             + synergy * p["hand_synergy"] + n_build * p["buildable_nodes"] + len(tiles) * p["num_tiles"]
             + n_hand * p["hand_resources"] + (p["discard_penalty"] if n_hand > 7 else 0)
             + lr_len * lr_factor + n_dev * p["hand_devs"] + knights * p["army_size"])
        if p["lr_always"]:
            v += lr_len * p["lr_always"]
        if p["opp_vp"]:
            opp = next(c for c in st.colors if c != color)
            v += st.player_state[f"{player_key(st, opp)}_VICTORY_POINTS"] * p["opp_vp"]
        if p["dev_bonus"] or p["army_bonus"]:
            v += n_dev * p["dev_bonus"] + knights * p["army_bonus"]
        if p["port2"] or p["port3"]:
            prod = {r: sample[f"EFFECTIVE_P0_{r}_PRODUCTION"] for r in RESOURCES}
            owned_set = set(owned)
            for res, nodes in st.board.map.port_nodes.items():
                if owned_set & nodes:
                    v += p["port3"] if res is None else p["port2"] * prod[res]
        return float(v)
    return fn
