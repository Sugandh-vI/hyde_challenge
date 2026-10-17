"""Opening features: describe one player's two setup settlements on a catanatron board.

Used both to generate training data (opengen.py) and at play time, so both see the same thing.
Settlements only, roads ignored.
"""
from catanatron.models.board import STATIC_GRAPH

RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
NAMES = ([f"pips_{r}" for r in RES] + ["pips_total", "n_res", "n_numbers"]
         + [f"port_{r}" for r in RES] + ["port_3", "port2_pips", "wood_brick", "wheat_ore",
                                          "sheep_wheat_ore", "best_open", "second_open", "n_open"])
N = len(NAMES)  # 21


def node_pips(board, n):
    return {r: p * 36 for r, p in board.map.node_production[n].items()}


def port_of(board):
    m = {}
    for res, nodes in board.map.port_nodes.items():
        for n in nodes:
            m[n] = res if res is not None else "3"
    return m


def open_nodes_after(board, occupied):
    """Nodes still legal for a settlement given occupied nodes (distance rule), ignoring roads."""
    blocked = set(occupied)
    for n in occupied:
        blocked.update(STATIC_GRAPH.neighbors(n))
    return [n for n in board.map.land_nodes if n not in blocked]


def side_features(board, mine, occupied, ports=None, open_nodes=None):
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
    """Feature row for the learned opening model: my side, their side, first-player flag."""
    occ = list(mine) + list(theirs)
    ports = port_of(board)
    open_nodes = set(open_nodes_after(board, occ))
    return (side_features(board, mine, occ, ports, open_nodes)
            + side_features(board, theirs, occ, ports, open_nodes) + [1.0 if i_am_first else 0.0])
