"""Cheap exact-ish model of F's setup settlement choice (no game copies).

F maximises 1e8*production(+variety) of the position after placing. Production = sum of tile
probabilities (robber starts on the desert, which produces nothing), variety = 4*0.02778 per
distinct resource produced. Other terms are >= 1e4x smaller, so they only break exact ties; on a tie
F keeps the FIRST action in list order (strict >), which we mimic by scanning candidates in order.
"""
from catanatron.models.board import STATIC_GRAPH

VAR = 4 * 2.778 / 100


def prod_value(board, nodes):
    tot = {}
    for n in nodes:
        for r, p in board.map.node_production[n].items():
            tot[r] = tot.get(r, 0.0) + p
    return sum(tot.values()) + VAR * sum(1 for v in tot.values() if v != 0)


def legal_nodes(board, occupied):
    blocked = set(occupied)
    for n in occupied:
        blocked.update(STATIC_GRAPH.neighbors(n))
    return [n for n in sorted(board.map.land_nodes) if n not in blocked]


def f_pick(board, f_nodes, occupied, candidates=None):
    best, bk = None, None
    for n in (candidates if candidates is not None else legal_nodes(board, occupied)):
        # F's next terms after production: reachable production (equal here), then the number
        # of tiles touched; values within float noise count as ties, then first in order wins
        k = (round(prod_value(board, list(f_nodes) + [n]), 6), len(board.map.adjacent_tiles[n]))
        if bk is None or k > bk:
            best, bk = n, k
    return best
