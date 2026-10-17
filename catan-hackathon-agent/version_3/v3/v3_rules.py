"""Hand-written overrides for the few decisions where a 1-ply value is clearly the wrong tool.

MOVE_ROBBER
    F scores the robber through our own value function, which mostly notices that our own
    production went down.  Here the robber goes where the opponent loses the most production and
    we lose the least, with extra credit for stealing a card and for denying the resource they are
    short of for their next build (their hand is exactly derivable from the bank and ours).

DISCARD_RESOURCE
    Half the hand is lost when a 7 is rolled.  We keep the cards that finish our nearest build:
    city first, then settlement, then a development card, then a road.

Each function returns an index into ``playable_actions``, or None when it does not apply.
"""

from catanatron.models.enums import CITY, SETTLEMENT, ActionType

RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]


def _owned(board, color):
    b = board.buildings_by_color[color]
    return list(b[SETTLEMENT]) + list(b[CITY])


def robber_index(world, actions, me, opp_resources=None, endgame=False):
    """Index of the MOVE_ROBBER action that hurts the opponent most and us least."""
    idxs = [i for i, a in enumerate(actions) if a.action_type == ActionType.MOVE_ROBBER]
    if not idxs:
        return None
    if len(idxs) == 1:
        return idxs[0]
    board = world.state.board
    try:
        opp = next(c for c in world.state.colors if c != me)
    except Exception:
        return idxs[0]

    hand = opp_resources or {}
    has_cards = sum(hand.values()) > 0
    deny = 1.6 if endgame else 1.0

    def missing(cost):
        return sum(max(0.0, v - hand.get(k, 0)) for k, v in cost.items())

    builds = [
        ({"WHEAT": 2, "ORE": 3}, 1.0),                            # city
        ({"WHEAT": 1, "SHEEP": 1, "BRICK": 1, "WOOD": 1}, 0.9),   # settlement
        ({"SHEEP": 1, "WHEAT": 1, "ORE": 1}, 0.8),                # development card
        ({"WOOD": 1, "BRICK": 1}, 0.7),                           # road
    ]
    closest = min(builds, key=lambda b: (missing(b[0]), -b[1]))
    need = {r for r, v in closest[0].items() if hand.get(r, 0) < v}

    best_i, best_s = idxs[0], -1e18
    for i in idxs:
        try:
            coord = actions[i].value[0]
            tile = board.map.tiles[coord]
        except Exception:
            continue
        opp_here = our_here = 0.0
        if getattr(tile, "number", None) is not None and tile.resource is not None:
            try:
                for n in tile.nodes.values():
                    try:
                        color, kind = board.buildings[n]
                    except Exception:
                        continue
                    mult = 2.0 if kind == CITY else 1.0
                    pips = board.map.node_production[n].get(tile.resource, 0.0) * 36.0 * mult
                    if color == opp:
                        opp_here += pips
                    elif color == me:
                        our_here += pips
            except Exception:
                pass
        s = (deny * opp_here - our_here
             + (1.0 if has_cards else 0.0)
             + (0.6 if tile.resource in need else 0.0))
        if s > best_s:
            best_i, best_s = i, s
    return best_i


def discard_index(actions, my_resources, my_devs=None):
    """Index of the DISCARD_RESOURCE action that damages our nearest build least."""
    idxs = [i for i, a in enumerate(actions) if a.action_type == ActionType.DISCARD_RESOURCE]
    if not idxs:
        return None
    if len(idxs) == 1:
        return idxs[0]
    have = {r: int((my_resources or {}).get(r, 0)) for r in RESOURCES}
    builds = [
        ({"WHEAT": 2, "ORE": 3}, 3.0),                                            # city
        ({"WHEAT": 1, "SHEEP": 1, "BRICK": 1, "WOOD": 1}, 2.0),                   # settlement
        ({"SHEEP": 1, "WHEAT": 1, "ORE": 1}, 1.4),                                # dev card
        ({"WOOD": 1, "BRICK": 1}, 1.0),                                           # road
    ]
    keep = {r: 0.0 for r in RESOURCES}
    for cost, weight in builds:
        miss = sum(max(0, v - have.get(r, 0)) for r, v in cost.items())
        closeness = 1.0 / (1.0 + miss)
        total = float(sum(cost.values()))
        for r, v in cost.items():
            if have.get(r, 0) < v:
                keep[r] += weight * closeness * (v / total)
            else:
                keep[r] += 0.15 * weight * closeness * (v / total)
    # a card we cannot use at all (and have plenty of) is the best one to throw away
    best_i, best_s = idxs[0], 1e18
    for i in idxs:
        try:
            r = actions[i].value
        except Exception:
            r = None
        s = keep.get(r, 0.0) - 0.08 * have.get(r, 0)
        if s < best_s:
            best_i, best_s = i, s
    return best_i
