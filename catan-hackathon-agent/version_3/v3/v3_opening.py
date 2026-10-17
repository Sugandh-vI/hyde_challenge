"""Opening: search our setup picks against a model of the opponent's replies.

Snake order with two players is  first, second, second, first.  At each of our picks we enumerate
our best candidate nodes, let a model place the opponent's remaining settlements, and score the
resulting four settlements with ``v3_openfeat.opening_score`` (ours minus theirs).  Choosing the
node that maximises the *final difference* is what makes this a counter-opening: we take the spot
whose follow-up position is best after the opponent has answered.

Opponent models (``OPP_MODEL``):
  "f_value"  the opponent plays F's own 1-ply rule on the real game object: for every legal
             BUILD_SETTLEMENT it executes the action on a copy and keeps the one that maximises
             ``base_fn`` for *their* colour, ties going to the earlier action, exactly like
             catanatron's ValueFunctionPlayer.  During the setup phase there is no hidden
             information, so this predicts F's placement exactly -- except that only the top
             ``opp_max_eval`` candidates by a cheap pip proxy are evaluated.
  "generic"  the opponent takes the node with the best generic opening score for their side.
             Robust against opponents that are not F.
  "mix"      the leaf score is the average of the two models' resulting positions.

The search never raises: any failure falls back to the best node by pips.
"""

import math
import time

from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import ActionType

import v3_openfeat as feat

DEFAULT_CFG = {
    "k1": 6,            # first-pick candidates we branch on
    "k2": 10,           # follow-up candidates per first pick
    "k2_mix": 6,        # follow-up candidates when two opponent models must be simulated
    "opp_max_eval": 12, # F-model: candidates evaluated exactly per reply (96/96 exact at 12)
    "coef": list(feat.DEFAULT_COEF),
}


def cfg_from(weights_opening):
    cfg = dict(DEFAULT_CFG)
    cfg["coef"] = list(feat.DEFAULT_COEF)
    if isinstance(weights_opening, dict):
        for k in ("k1", "k2", "k2_mix", "opp_max_eval"):
            if k in weights_opening:
                try:
                    cfg[k] = int(weights_opening[k])
                except Exception:
                    pass
        coef = weights_opening.get("coef")
        if isinstance(coef, (list, tuple)) and len(coef) == feat.N_FEATURES:
            cfg["coef"] = [float(x) for x in coef]
    return cfg


# ---------------------------------------------------------------------------
# helpers on a live game copy
# ---------------------------------------------------------------------------
def _settlement_actions(game):
    return [a for a in game.playable_actions if a.action_type == ActionType.BUILD_SETTLEMENT]


def _place(game, node):
    """Copy of `game` with a settlement on `node` and any legal road played."""
    g = game.copy()
    action = None
    for a in g.playable_actions:
        if a.action_type == ActionType.BUILD_SETTLEMENT and a.value == node:
            action = a
            break
    if action is None:
        return None
    g.execute(action)
    for a in g.playable_actions:
        if a.action_type == ActionType.BUILD_ROAD:
            g.execute(a)
            break
    return g


def _avail(game):
    return [a.value for a in _settlement_actions(game)]


def _proxy_key(board, node):
    p = feat.node_pips(board, node)
    return sum(p.values()) + 2.0 * len(p)


def _top(nodes, board, k):
    return sorted(nodes, key=lambda n: (_proxy_key(board, n), -n), reverse=True)[:k]


# ---------------------------------------------------------------------------
# opponent models
# ---------------------------------------------------------------------------
def _f_pick(game, opp_color, cfg, base_fn):
    """F's setup pick: argmax of base_fn after executing the settlement, first max wins."""
    acts = _settlement_actions(game)
    if not acts:
        return None
    if cfg["opp_max_eval"] and len(acts) > cfg["opp_max_eval"]:
        keep = set(_top([a.value for a in acts], game.state.board, cfg["opp_max_eval"]))
        acts = [a for a in acts if a.value in keep]          # engine order preserved
    best, best_v = None, None
    for a in acts:
        g = game.copy()
        try:
            g.execute(a)
        except Exception:
            continue
        v = base_fn(g, opp_color)
        if best_v is None or v > best_v:
            best, best_v = a.value, v
    return best


def _generic_pick(game, mine, theirs, coef):
    """Best node by our generic opening score, from the opponent's point of view."""
    acts = _settlement_actions(game)
    if not acts:
        return None
    board = game.state.board
    best, best_v = None, None
    for a in acts:
        s = feat.opening_score(board, list(theirs) + [a.value], list(mine), coef)
        if best_v is None or s > best_v:
            best, best_v = a.value, s
    return best


def _reply_tree(game, mine, theirs, opp_color, model, cfg, base_fn, n):
    """Apply n opponent placements in a row.  Returns (nodes, final_game) or (None, None)."""
    nodes = []
    cur = game
    for _ in range(n):
        node = None
        if model == "f_value":
            node = _f_pick(cur, opp_color, cfg, base_fn)
        elif model == "generic":
            node = _generic_pick(cur, mine if n == 1 else mine, theirs, cfg["coef"])
        if node is None:
            return None, None
        nxt = _place(cur, node)
        if nxt is None:
            return None, None
        nodes.append(node)
        cur = nxt
    return nodes, cur


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def choose_settlement(world, cands, mine, theirs, me, model, cfg, base_fn, deadline=None):
    """Return the settlement node to build now, or None.

    world       a catanatron Game consistent with the observation (setup phase)
    cands       node ids legal for us right now (from the action list)
    mine/theirs node ids already built
    me          our colour
    model       "f_value" | "generic" | "mix"
    base_fn     F's value function, used by the "f_value" model
    deadline    time.process_time() value; the search returns its best-so-far at the deadline
    """
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    board = world.state.board
    coef = cfg["coef"]
    mine, theirs = list(mine), list(theirs)
    try:
        opp_color = next(c for c in world.state.colors if c != me)
    except Exception:
        return _top(cands, board, 1)[0]

    def over():
        return deadline is not None and time.process_time() > deadline

    def leaf(mine2, theirs2):
        return feat.opening_score(board, mine2, theirs2, coef)

    models = ("f_value", "generic") if model == "mix" else (model,)
    k2 = cfg["k2_mix"] if model == "mix" else cfg["k2"]
    best, best_v = None, -math.inf

    # ---- case A: we place first (picks 1 and 4) -------------------------------------------
    if len(mine) == 0 and len(theirs) == 0:
        for c1 in _top(cands, board, cfg["k1"]):
            if best is not None and over():
                break
            g1 = _place(world, c1)
            if g1 is None:
                continue
            trees = []
            for m in models:
                nodes, gend = _reply_tree(g1, [c1], [], opp_color, m, cfg, base_fn, 2)
                if nodes is not None and gend is not None:
                    trees.append((nodes, _avail(gend)))
            if not trees:
                continue
            for c2 in _top(sorted({n for _nodes, rest in trees for n in rest}), board, k2):
                if best is not None and over():
                    break
                vals = []
                for nodes, _rest in trees:
                    if c2 in nodes:
                        continue
                    vals.append(leaf([c1, c2], theirs + nodes))
                if vals:
                    v = sum(vals) / len(vals)
                    if v > best_v:
                        best, best_v = c1, v
        if best is not None:
            return best

    # ---- case B: we place second, first pick (picks 2 and 3) ------------------------------
    elif len(mine) == 0 and len(theirs) >= 1:
        for c1 in _top(cands, board, cfg["k1"]):
            if best is not None and over():
                break
            g1 = _place(world, c1)
            if g1 is None:
                continue
            rest = _avail(g1)
            if not rest:
                continue
            for c2 in _top(rest, board, k2):
                if best is not None and over():
                    break
                g2 = _place(g1, c2)
                if g2 is None:
                    continue
                mine2 = mine + [c1, c2]
                vals = []
                for m in models:
                    nodes, _g = _reply_tree(g2, mine2, theirs, opp_color, m, cfg, base_fn, 1)
                    if nodes:
                        vals.append(leaf(mine2, theirs + nodes))
                if vals:
                    v = sum(vals) / len(vals)
                    if v > best_v:
                        best, best_v = c1, v
        if best is not None:
            return best

    # ---- case C: we place second, second pick (pick 3, one reply left) --------------------
    elif len(mine) == 1 and len(theirs) >= 1:
        for c2 in _top(cands, board, k2):
            if best is not None and over():
                break
            g2 = _place(world, c2)
            if g2 is None:
                continue
            mine2 = mine + [c2]
            vals = []
            for m in models:
                nodes, _g = _reply_tree(g2, mine2, theirs, opp_color, m, cfg, base_fn, 1)
                if nodes:
                    vals.append(leaf(mine2, theirs + nodes))
            if vals:
                v = sum(vals) / len(vals)
                if v > best_v:
                    best, best_v = c2, v
        if best is not None:
            return best

    # ---- case D: our last pick, nothing to predict ---------------------------------------
    else:
        for c2 in _top(cands, board, cfg["k2"]):
            v = leaf(mine + [c2], theirs)
            if v > best_v:
                best, best_v = c2, v
        if best is not None:
            return best

    # ---- fallback ------------------------------------------------------------------------
    return _top(cands, board, 1)[0]


def choose_road(world, actions, mine, theirs, cfg):
    """Initial road: point it at the best still-open settlement one step beyond the road."""
    if not actions:
        return 0
    board = world.state.board
    coef = cfg["coef"]
    occ = set(mine) | set(theirs)
    open_nodes = set(feat.open_nodes_after(board, list(occ)))
    best_i, best_s = 0, -math.inf
    for i, a in enumerate(actions):
        try:
            u, v = a.value
        except Exception:
            continue
        far = v if u in mine else (u if v in mine else v)
        spots = [n for n in STATIC_GRAPH.neighbors(far) if n in open_nodes and n not in occ]
        s = max((feat.opening_score(board, list(mine) + [n], list(theirs), coef)
                 for n in spots), default=-1e18)
        if s > best_s:
            best_i, best_s = i, s
    return best_i
