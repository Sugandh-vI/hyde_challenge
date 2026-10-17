"""Counter-opening: search our setup picks against F's predicted replies (fmodel.f_pick).

Snake order is first, second, second, first. At each of our picks we enumerate our remaining
picks, play F's replies with f_pick, and score the final 4 settlements with a P(win) evaluator.
Evaluator = learned logistic on openfeat.game_features if weights exist, else a hand score.
"""
import json
import math
import os

from fmodel import f_pick, legal_nodes
from openfeat import game_features, node_pips

_W = None
_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "open_logreg.json")
if os.path.exists(_PATH):
    with open(_PATH) as fh:
        _d = json.load(fh)
    _W = (_d["mean"], _d["std"], _d["coef"], _d["bias"])


def evaluate(board, mine, theirs, i_first):
    x = game_features(board, mine, theirs, i_first)
    if _W is None:
        # hand fallback: pips + 3 per resource type + balance, mine minus theirs
        h = len(x) // 2
        def side(f):
            return f[5] + 3 * f[6] + f[17] + f[18]
        return side(x[:h]) - side(x[h:2 * h])
    mean, std, coef, bias = _W
    return bias + sum(c * (v - m) / s for c, v, m, s in zip(coef, x, mean, std))


def rank(board, nodes, k):
    def quick(n):
        p = node_pips(board, n)
        return sum(p.values()) + 2 * len(p)
    return sorted(nodes, key=quick, reverse=True)[:k]


def choose(board, cands, mine, theirs, i_first, k1=6, k2=10):
    """Return the settlement node to build now. cands = legal nodes from the action list."""
    occ = list(mine) + list(theirs)
    if len(mine) == 0 and len(theirs) == 0:              # first player, pick 1
        best, bv = None, -math.inf
        for c1 in rank(board, cands, k1):
            f1 = f_pick(board, [], occ + [c1])
            f2 = f_pick(board, [f1], occ + [c1, f1])
            rest = legal_nodes(board, [c1, f1, f2])
            v = max((evaluate(board, [c1, c2], [f1, f2], True) for c2 in rank(board, rest, k2)),
                    default=-math.inf)
            if v > bv:
                best, bv = c1, v
        return best
    if len(mine) == 0 and len(theirs) == 1:              # second player, pick 1 (then pick 2)
        best, bv = None, -math.inf
        for c1 in rank(board, cands, k1):
            v = -math.inf
            for c2 in rank(board, legal_nodes(board, occ + [c1]), k2):
                f2 = f_pick(board, list(theirs), occ + [c1, c2])
                v = max(v, evaluate(board, [c1, c2], list(theirs) + [f2], False))
            if v > bv:
                best, bv = c1, v
        return best
    if len(mine) == 1 and len(theirs) == 1:              # second player, pick 2
        best, bv = None, -math.inf
        for c2 in rank(board, cands, 2 * k2):
            f2 = f_pick(board, list(theirs), occ + [c2])
            v = evaluate(board, list(mine) + [c2], list(theirs) + [f2], False)
            if v > bv:
                best, bv = c2, v
        return best
    # first player, last pick: nothing left to predict
    return max(rank(board, cands, 3 * k2), key=lambda c: evaluate(board, list(mine) + [c], list(theirs), True))
