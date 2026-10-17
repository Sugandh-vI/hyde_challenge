"""Value function for version_3: F's base_fn plus a small set of extra terms.

With every extra weight at 0 this function is numerically identical to F's value function
(``catanatron.players.value.get_value_fn("base_fn", DEFAULT_WEIGHTS)``).  ``tools/check_value.py``
verifies that on real positions.

Extra terms (all weights live in weights.json -> "extra"):

  lr_always    longest-road length, always on.  F only counts it when no buildable node is left
               and otherwise multiplies it by 0.1, so in practice 1 unit of road is worth ~1 —
               smaller than `num_tiles` (1) and `hand_resources` (1).  A larger always-on term
               acts as a cheap tie-breaker that prefers continuing a road.
  port2        owned 2:1 port, weighted by our production of that resource (a port is worth more
               the more of that resource we actually produce).
  port3        owned 3:1 port, flat.
  hidden_vp    our own unplayed victory-point cards, which we can see (obs.my_devs) but base_fn
               cannot: it scores public VPs only.  Weight is in units of public_vps.
  opp_vp       the opponent's public VPs (negative weight = play more defensively when they are
               close to winning).  Off by default, swept.
  dist_city / dist_settlement / dist_dev
               "cards still missing" terms: less than 2 wheat + 3 ore for a city, etc.  Off by
               default (hand_synergy already covers part of this), swept.

Everything is computed in one pass so the cost is close to base_fn's own.
"""

import json
import os

from catanatron.features import (build_production_features, reachability_features,
                                 resource_hand_features)
from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT
from catanatron.players.value import DEFAULT_WEIGHTS, value_production
from catanatron.state_functions import (get_longest_road_length, get_played_dev_cards,
                                        player_key, player_num_dev_cards,
                                        player_num_resource_cards)

#: Extra weights.  DEFAULTS ARE A STARTING POINT, not tuned: sweep.py searches around them.
DEFAULT_EXTRAS = {
    "lr_always": 1000.0,      # tie-breaker well above num_tiles (1) and below production (1e8)
    "port2": 2.0e7,           # ~ one pip of production when we produce the port's resource
    "port3": 1.0e6,
    "hidden_vp": 1.0,         # in units of public_vps: our own VP cards count in full
    "opp_vp": 0.0,
    "dist_city": 0.0,
    "dist_settlement": 0.0,
    "dist_dev": 0.0,
}

_PROD = build_production_features(True)


def _folder():
    return os.path.dirname(os.path.abspath(__file__))


def load_config(folder=None):
    """Return {"base": 13 weights, "extra": ..., "opening": {...}}.

    ``weights_best.json`` wins over ``weights.json`` when both exist (sweep.py writes the former).
    """
    folder = folder or _folder()
    base = dict(DEFAULT_WEIGHTS)
    extras = dict(DEFAULT_EXTRAS)
    opening = {}
    for name in ("weights_best.json", "weights.json"):
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict):
            for k, v in (data.get("base") or {}).items():
                if k in DEFAULT_WEIGHTS:
                    try:
                        base[k] = float(v)
                    except Exception:
                        pass
            for k, v in (data.get("extra") or {}).items():
                if k in DEFAULT_EXTRAS:
                    try:
                        extras[k] = float(v)
                    except Exception:
                        pass
            if isinstance(data.get("opening"), dict):
                opening.update(data["opening"])
        break
    return {"base": base, "extra": extras, "opening": opening}


def make_value_fn(base_weights=None, extras=None):
    """Build ``fn(game, color) -> float``; extras all zero reproduces F exactly."""
    p = dict(DEFAULT_WEIGHTS)
    p.update(base_weights or {})
    e = dict(DEFAULT_EXTRAS)
    e.update(extras or {})
    vp = p["public_vps"]

    def fn(game, color):
        st = game.state
        sample = _PROD(game, color)
        production = value_production(sample, "P0")
        enemy_production = value_production(sample, "P1", False)

        key = player_key(st, color)
        ps = st.player_state
        lr_len = get_longest_road_length(st, color)

        reach = reachability_features(game, color, 2)
        r0 = sum(reach["P0_0_ROAD_REACHABLE_" + r] for r in RESOURCES)
        r1 = sum(reach["P0_1_ROAD_REACHABLE_" + r] for r in RESOURCES)

        hand = resource_hand_features(game, color)
        d_city = (max(2 - hand["P0_WHEAT_IN_HAND"], 0)
                  + max(3 - hand["P0_ORE_IN_HAND"], 0)) / 5.0
        d_sett = (max(1 - hand["P0_WHEAT_IN_HAND"], 0) + max(1 - hand["P0_SHEEP_IN_HAND"], 0)
                  + max(1 - hand["P0_BRICK_IN_HAND"], 0) + max(1 - hand["P0_WOOD_IN_HAND"], 0)) / 4.0
        synergy = (2 - d_city - d_sett) / 2

        n_hand = player_num_resource_cards(st, color)
        discard_penalty = p["discard_penalty"] if n_hand > 7 else 0

        buildings = st.buildings_by_color[color]
        owned = buildings[SETTLEMENT] + buildings[CITY]
        tiles = set()
        for n in owned:
            tiles.update(st.board.map.adjacent_tiles[n])

        n_build = len(st.board.buildable_node_ids(color))
        lr_factor = p["longest_road"] if n_build == 0 else 0.1
        n_dev = player_num_dev_cards(st, color)
        knights = get_played_dev_cards(st, color, "KNIGHT")

        v = (ps[key + "_VICTORY_POINTS"] * vp
             + production * p["production"]
             + enemy_production * p["enemy_production"]
             + r0 * p["reachable_production_0"]
             + r1 * p["reachable_production_1"]
             + synergy * p["hand_synergy"]
             + n_build * p["buildable_nodes"]
             + len(tiles) * p["num_tiles"]
             + n_hand * p["hand_resources"]
             + discard_penalty
             + lr_len * lr_factor
             + n_dev * p["hand_devs"]
             + knights * p["army_size"])

        if e["lr_always"]:
            v += lr_len * e["lr_always"]
        if e["hidden_vp"]:
            hidden = ps[key + "_ACTUAL_VICTORY_POINTS"] - ps[key + "_VICTORY_POINTS"]
            if hidden:
                v += hidden * vp * e["hidden_vp"]
        if e["opp_vp"]:
            try:
                opp = next(c for c in st.colors if c != color)
                v += ps[player_key(st, opp) + "_VICTORY_POINTS"] * e["opp_vp"]
            except Exception:
                pass
        if e["port2"] or e["port3"]:
            owned_set = set(owned)
            for res, nodes in st.board.map.port_nodes.items():
                if owned_set.intersection(nodes):
                    if res is None:
                        if e["port3"]:
                            v += e["port3"]
                    elif e["port2"]:
                        v += e["port2"] * sample["EFFECTIVE_P0_" + res + "_PRODUCTION"]
        if e["dist_city"]:
            v -= e["dist_city"] * (max(2 - hand["P0_WHEAT_IN_HAND"], 0)
                                   + max(3 - hand["P0_ORE_IN_HAND"], 0))
        if e["dist_settlement"]:
            v -= e["dist_settlement"] * (max(1 - hand["P0_WHEAT_IN_HAND"], 0)
                                         + max(1 - hand["P0_SHEEP_IN_HAND"], 0)
                                         + max(1 - hand["P0_BRICK_IN_HAND"], 0)
                                         + max(1 - hand["P0_WOOD_IN_HAND"], 0))
        if e["dist_dev"]:
            v -= e["dist_dev"] * (max(1 - hand["P0_SHEEP_IN_HAND"], 0)
                                  + max(1 - hand["P0_WHEAT_IN_HAND"], 0)
                                  + max(1 - hand["P0_ORE_IN_HAND"], 0))
        return float(v)

    return fn
