"""Opening features: describe one side's setup settlements on a catanatron board.

Used by both the opening search (v3_opening) and, at evaluation time, to score a candidate
placement.  Settlements only; roads are handled separately.

A "pip" here is the textbook one: 5 for a 6/8, 4 for 5/9, 3 for 4/10, 2 for 3/11, 1 for 2/12,
0 for the desert. ``board.map.node_production`` stores probabilities (pips / 36).
"""

from catanatron.models.board import STATIC_GRAPH

RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]

#: Names of the 21 features produced by ``side_features`` (order matters).
NAMES = (
    [f"pips_{r}" for r in RES]
    + ["pips_total", "n_res", "n_numbers"]
    + [f"port_{r}" for r in RES]
    + ["port_3", "port2_pips", "wood_brick", "wheat_ore", "sheep_wheat_ore",
       "best_open", "second_open", "n_open"]
)
N_FEATURES = len(NAMES)  # 21

#: Default coefficients for ``opening_score`` (one per feature, see NAMES).
#: Chosen so that pips dominate, diversity and ports break ties, and open expansion room is a
#: small bonus.  All of them are tunable in weights.json ("opening" -> "coef").
DEFAULT_COEF = [
    1.0, 1.0, 1.0, 1.0, 1.0,      # pips per resource
    0.0,                          # pips_total (redundant with the 5 above)
    1.5,                          # n_res: diversity is worth a lot in a 2-player game
    0.3,                          # n_numbers: distinct numbers, mild
    0.8, 0.8, 0.8, 0.8, 0.8,      # 2:1 port flags
    0.4,                          # 3:1 port flag
    0.25,                         # production of the resources we hold a port for
    1.0,                          # wood+brick balance (early expansion)
    0.6,                          # wheat+ore balance (cities)
    0.5,                          # sheep+wheat+ore (dev cards)
    0.8,                          # best open follow-up spot
    0.4,                          # second best open follow-up spot
    0.0,                          # number of open follow-up spots
]


def node_pips(board, node):
    """{resource: pips} produced by a settlement on ``node`` (desert excluded)."""
    return {r: p * 36.0 for r, p in board.map.node_production[node].items()}


def port_of(board):
    """{node: resource name or '3'} for every port node on the board."""
    m = {}
    for res, nodes in board.map.port_nodes.items():
        for n in nodes:
            m[n] = res if res is not None else "3"
    return m


def open_nodes_after(board, occupied):
    """Land nodes still legal for a settlement given ``occupied`` (distance rule), no roads."""
    blocked = set(occupied)
    for n in occupied:
        blocked.update(STATIC_GRAPH.neighbors(n))
    return [n for n in board.map.land_nodes if n not in blocked]


def side_features(board, mine, occupied, ports=None, open_nodes=None):
    """21 numbers describing one player's settlement set (see NAMES)."""
    ports = port_of(board) if ports is None else ports
    pips = {r: 0.0 for r in RES}
    numbers = set()
    for n in mine:
        for r, p in node_pips(board, n).items():
            pips[r] += p
        for t in board.map.adjacent_tiles[n]:
            if t.number is not None:
                numbers.add(t.number)
    f = [pips[r] for r in RES]
    f += [sum(pips.values()), sum(1 for r in RES if pips[r] > 0), len(numbers)]
    my_ports = {ports[n] for n in mine if n in ports}
    f += [1.0 if r in my_ports else 0.0 for r in RES] + [1.0 if "3" in my_ports else 0.0]
    f.append(sum(pips[r] for r in RES if r in my_ports))
    f += [min(pips["WOOD"], pips["BRICK"]), min(pips["WHEAT"], pips["ORE"]),
          min(pips["SHEEP"], pips["WHEAT"], pips["ORE"])]
    if open_nodes is None:
        open_nodes = set(open_nodes_after(board, occupied))
    near = set()
    for n in mine:
        for a in STATIC_GRAPH.neighbors(n):
            for b in STATIC_GRAPH.neighbors(a):
                if b in open_nodes:
                    near.add(b)
    vals = sorted((sum(node_pips(board, n).values()) for n in near), reverse=True)
    f += [vals[0] if vals else 0.0, vals[1] if len(vals) > 1 else 0.0, float(len(vals))]
    return f


def game_features(board, mine, theirs, i_am_first):
    """Concatenated features of both sides plus a first-player flag (43 numbers)."""
    occ = list(mine) + list(theirs)
    ports = port_of(board)
    open_nodes = set(open_nodes_after(board, occ))
    return (side_features(board, mine, occ, ports, open_nodes)
            + side_features(board, theirs, occ, ports, open_nodes)
            + [1.0 if i_am_first else 0.0])


def opening_score(board, mine, theirs, coef):
    """Our opening value minus theirs.  ``coef`` has 21 entries (see DEFAULT_COEF)."""
    occ = list(mine) + list(theirs)
    ports = port_of(board)
    open_nodes = set(open_nodes_after(board, occ))
    def side(nodes):
        f = side_features(board, nodes, occ, ports, open_nodes)
        return sum(c * x for c, x in zip(coef, f))
    return side(mine) - side(theirs)


def quick_node_rank(board, nodes, k):
    """Cheap pre-rank of settlement nodes: pips plus a diversity bonus (deterministic order)."""
    def key(n):
        p = node_pips(board, n)
        return (sum(p.values()) + 2.0 * len(p), -n)
    return sorted(nodes, key=key, reverse=True)[:k]
