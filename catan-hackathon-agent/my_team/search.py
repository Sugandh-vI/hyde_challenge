"""Search helpers for agent v2.

Everything here is searched on a *sampled world* (a complete catanatron Game that agrees with
what the agent can see).  The agent itself never constructs Actions: all candidate actions are
taken from the game's own playable-action list, and chance nodes are expanded with the engine's
``execute_spectrum`` helper.

Budgeting
---------
``obs.cpu_left`` is the exact number of seconds the harness still allows for this game, this
decision's 12.5 ms increment included.  We spend at most ``compute_budget(...)`` seconds of
``time.process_time()`` per decision and leave ``RESERVE_S`` for the rest of the game, so the
clock can never run out (which would make the harness play random moves).
"""

import json
import os
import time

from catanatron.models.enums import ActionType
from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn
from catanatron.state_functions import get_player_freqdeck, player_key
from catanatron.players.tree_search_utils import execute_spectrum

# ---------------------------------------------------------------------------
# Knobs (all in one place, easy to tune / report)
# ---------------------------------------------------------------------------
K_WORLDS = 1            # sampled worlds averaged at depth 1 (1 = F's information, cheap)
BEAM_WIDTH = 3          # children kept per node when deepening (B ~ 3-5)
ROOT_BEAM = 6           # root actions that get deepened (the rest keep their depth-1 value)
MAX_DEPTH = 3           # max number of our own actions in a same-turn line
RESERVE_S = 0.25        # CPU kept in reserve for the rest of the game (no game over the clock)
CAP_NORMAL_S = 0.025    # per-decision cap outside the opening placement
CAP_OPENING_S = 0.060   # per-decision cap during the initial placement (many legal moves)
EST_DECISIONS_LEFT = 30 # denominator of the per-decision budget
OPENING_TURNS = 4       # first four decisions may think longer
SAFETY = 0.80           # fraction of the budget the internal deadline uses
WIN = float("inf")

CHANCE_TYPES = frozenset(
    [ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD]
)


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------
def action_key(action):
    """Hashable identity of an action, used to map world actions back to indices."""
    try:
        return (action.action_type, action.value)
    except Exception:
        return (action.action_type, str(action.value))


def is_chance(action):
    """True if executing `action` has a random outcome that must be averaged over."""
    at = action.action_type
    if at in CHANCE_TYPES:
        return True
    if at == ActionType.MOVE_ROBBER:
        try:
            value = action.value
            victim = value[1] if len(value) >= 2 else None
            return victim is not None
        except Exception:
            return False
    return False


def _victim_color(action):
    try:
        return action.value[1]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Weights / value function
# ---------------------------------------------------------------------------
def load_weights(folder=None):
    """weights.json next to the agent, falling back to catanatron's DEFAULT_WEIGHTS."""
    weights = dict(DEFAULT_WEIGHTS)
    folder = folder or os.path.dirname(os.path.abspath(__file__))
    try:
        path = os.path.join(folder, "weights.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                for k, v in loaded.items():
                    try:
                        weights[k] = float(v)
                    except Exception:
                        pass
    except Exception:
        pass
    return weights


def make_value_fn(weights=None):
    """base_fn plus our hidden victory-point cards.

    base_fn scores ``P?_VICTORY_POINTS``, i.e. public points only: a victory-point card we hold
    is invisible to it.  We know our own development cards exactly, so ``ACTUAL_VICTORY_POINTS``
    is known for our seat, and adding ``(actual - public) * public_vps`` makes the agent value
    its own hidden points (and therefore play a winning move when it has one).
    """
    weights = weights or load_weights()
    base = get_value_fn("base_fn", weights)
    vp_weight = float(weights.get("public_vps", 3e14))

    def value_fn(game, me):
        v = base(game, me)
        try:
            key = player_key(game.state, me)
            ps = game.state.player_state
            hidden = ps[f"{key}_ACTUAL_VICTORY_POINTS"] - ps[f"{key}_VICTORY_POINTS"]
            if hidden:
                v += hidden * vp_weight
        except Exception:
            pass
        return v

    return value_fn


# ---------------------------------------------------------------------------
# Time budget
# ---------------------------------------------------------------------------
def compute_budget(cpu_left, is_opening):
    """Seconds of CPU we may spend on this decision."""
    try:
        cpu_left = float(cpu_left) if cpu_left is not None else 1.0
    except Exception:
        cpu_left = 1.0
    if is_opening:
        budget = min(CAP_OPENING_S, max(0.010, cpu_left * 0.16))
    else:
        avail = max(0.0, cpu_left - RESERVE_S)
        budget = avail / float(EST_DECISIONS_LEFT)
        if budget > CAP_NORMAL_S:
            budget = CAP_NORMAL_S
        if budget < 0.0015:
            budget = min(0.0015, max(cpu_left * 0.25, 0.0002))
    return max(0.0, min(budget, cpu_left * 0.75))


# ---------------------------------------------------------------------------
# Candidate pruning
# ---------------------------------------------------------------------------
def _pruned_actions(game):
    """Candidates for search: every legal action except redundant maritime trades.

    catanatron's own ``list_prunned_actions`` also keeps a single robber tile chosen by a
    production heuristic.  Measurements (EXPERIMENTS.md, E3/E4) showed that this robber pruning
    costs ~10 points of win rate, far more than it saves in CPU, so we keep every robber move
    and only remove trades that cannot be better than another legal trade.
    """
    try:
        actions = list(game.playable_actions)
    except Exception:
        return []
    try:
        port_resources = game.state.board.get_player_port_resources(game.state.current_color())
        has_three_to_one = None in port_resources
    except Exception:
        has_three_to_one = False
    out = []
    seen = set()
    for a in actions:
        if a.action_type == ActionType.MARITIME_TRADE:
            try:
                key = action_key(a)
                if key in seen:
                    continue
                seen.add(key)
                # a 4:1 trade is strictly dominated by a 3:1 trade for the same resource pair
                if has_three_to_one and len(a.value) >= 4 and a.value[3] is not None:
                    continue
            except Exception:
                pass
        out.append(a)
    return out


def _candidate_indices(world, playable_actions):
    """Indices in `playable_actions` that survive catanatron's pruning.

    The pruned list from the sampled world is mapped back by (action_type, value).  The order of
    playable_actions is preserved, so ties keep the same winner as F would pick.
    """
    try:
        pruned = _pruned_actions(world)
        allowed = {action_key(a) for a in pruned}
        idxs = [i for i, a in enumerate(playable_actions) if action_key(a) in allowed]
        if idxs:
            return idxs
    except Exception:
        pass
    return list(range(len(playable_actions)))


# ---------------------------------------------------------------------------
# Chance nodes
# ---------------------------------------------------------------------------
def _chance_outcomes(game, action):
    """[(outcome_game, probability), ...] for a chance action, or None if unavailable.

    Uses the engine's spectrum expansion.  For a robber steal the engine assigns 1/5 to each
    resource; the victim's hand is public in 2-player Catan, so the real probabilities are the
    card counts over the hand size, and we correct them here.
    """
    try:
        outcomes = execute_spectrum(game, action)
    except Exception:
        return None
    if not outcomes:
        return None
    try:
        if action.action_type == ActionType.MOVE_ROBBER and is_chance(action):
            victim = _victim_color(action)
            hand = get_player_freqdeck(game.state, victim)
            total = float(sum(hand))
            if total <= 0:
                return outcomes
            resources = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
            fixed = []
            for outcome_game, _p in outcomes:
                card = None
                try:
                    card = outcome_game.state.action_records[-1].result
                except Exception:
                    pass
                if card in resources:
                    fixed.append((outcome_game, hand[resources.index(card)] / total))
                else:
                    fixed.append((outcome_game, 0.0))
            total_p = sum(p for _g, p in fixed)
            if total_p > 0:
                return [(g, p / total_p) for g, p in fixed]
    except Exception:
        pass
    return outcomes


def _chance_value(outcomes, value_fn, me):
    """Probability-weighted value. Also returns True if one outcome wins immediately."""
    total = 0.0
    weight = 0.0
    for game, p in outcomes:
        try:
            if game.winning_color() == me:
                return WIN, True
            total += p * value_fn(game, me)
            weight += p
        except Exception:
            continue
    if weight <= 0.0:
        return None, False
    return total / weight, False


# ---------------------------------------------------------------------------
# One-step evaluation
# ---------------------------------------------------------------------------
def eval_action(game, action, value_fn, me):
    """(value, successor_or_None, chance_outcomes_or_None, won).

    value is the one-step value F would use, except that chance nodes are averaged over their
    outcomes and a move that wins the game returns WIN.
    """
    if is_chance(action):
        outcomes = _chance_outcomes(game, action)
        if outcomes is not None:
            value, won = _chance_value(outcomes, value_fn, me)
            if won:
                return WIN, None, outcomes, True
            if value is not None:
                return value, None, outcomes, False
    succ = game.copy()
    succ.execute(action)
    if succ.winning_color() is not None and succ.winning_color() == me:
        return WIN, succ, None, True
    return value_fn(succ, me), succ, None, False


# ---------------------------------------------------------------------------
# Same-turn search
# ---------------------------------------------------------------------------
def _over(deadline):
    return time.process_time() >= deadline


def _continuation_value(game, depth, deadline, value_fn, me, beam):
    """Best value reachable from `game` using at most `depth` more of our own actions."""
    try:
        if depth <= 0 or _over(deadline):
            return value_fn(game, me)
        if game.winning_color() is not None:
            if game.winning_color() == me:
                return WIN
            return value_fn(game, me)
        if game.state.current_color() != me:
            return value_fn(game, me)
    except Exception:
        try:
            return value_fn(game, me)
        except Exception:
            return 0.0

    try:
        actions = _pruned_actions(game)
    except Exception:
        return value_fn(game, me)
    if not actions:
        return value_fn(game, me)

    base = value_fn(game, me)          # value of stopping here (≈ END_TURN)
    scored = []
    for action in actions:
        if _over(deadline):
            return base
        try:
            v, succ, outcomes, won = eval_action(game, action, value_fn, me)
        except Exception:
            continue
        if won:
            return WIN
        scored.append((v, action, succ, outcomes))
    if not scored:
        return base

    scored.sort(key=lambda x: x[0], reverse=True)
    best = base
    for v, action, succ, outcomes in scored[:beam]:
        if action.action_type == ActionType.END_TURN or succ is None:
            cand = v
        else:
            try:
                if succ.state.current_color() != me or succ.winning_color() is not None:
                    cand = v
                else:
                    cand = _continuation_value(succ, depth - 1, deadline, value_fn, me, beam)
            except Exception:
                cand = v
        if cand > best:
            best = cand
    return best


def choose_action(worlds, me, playable_actions, value_fn, budget_s,
                  beam=BEAM_WIDTH, root_beam=ROOT_BEAM, max_depth=MAX_DEPTH):
    """Return (index, value) of the best action in `playable_actions`.

    Depth 1 is F's one-step evaluation on `worlds[0]` (averaged over worlds when K > 1) and is
    always completed first, so a starved search falls back to F, never to something worse.
    Deeper levels only deepen the `root_beam` actions with the best depth-1 values; each line is
    scored at END_TURN / a change of turn, and chance nodes are probability-weighted.
    """
    if not playable_actions:
        return 0, 0.0
    if not worlds:
        return 0, 0.0

    start = time.process_time()
    deadline = start + max(0.0, budget_s) * SAFETY
    # The depth-1 pass (F's own evaluation) is ALWAYS completed.  Cutting it short returns the
    # best of an arbitrary prefix of the action list, which is much worse than F: measured at
    # K=2 worlds, a truncated depth-1 pass scored 0.125 against F (EXPERIMENTS.md E8).  Depth 1
    # costs about what F costs, and the harness allowance is six times F's budget, so completing
    # it is always affordable; only the optional deepening below is deadline-limited.
    world = worlds[0]

    try:
        idxs = _candidate_indices(world, playable_actions)
    except Exception:
        idxs = list(range(len(playable_actions)))

    # ---- depth 1: exactly F, on the (possibly averaged) sampled worlds -------------------
    scored = []
    for i in idxs:
        action = playable_actions[i]
        try:
            if len(worlds) == 1 or is_chance(action):
                v, _succ, _out, won = eval_action(world, action, value_fn, me)
            else:
                vals = []
                won = False
                for w in worlds:
                    vv, _s, _o, w_won = eval_action(w, action, value_fn, me)
                    won = won or w_won
                    vals.append(vv)
                v = sum(vals) / len(vals) if vals else 0.0
                if won:
                    v = WIN
        except Exception:
            continue
        if won:
            return i, WIN
        scored.append((v, i))

    if not scored:
        # Every action failed on the sampled world: fall back to a cheap one-step scan.
        best_i, best_v = 0, -float("inf")
        for i, a in enumerate(playable_actions):
            try:
                _v, _s, _o, _w = eval_action(world, a, value_fn, me)
                if _v > best_v:
                    best_i, best_v = i, _v
            except Exception:
                continue
        return best_i, best_v

    scored.sort(key=lambda x: x[0], reverse=True)
    best_val, best_idx = scored[0]

    # ---- deeper: iterative deepening over our own same-turn actions ----------------------
    if budget_s > 0.002:
        for depth in range(2, max_depth + 1):
            if _over(deadline):
                break
            improved_val, improved_idx = -float("inf"), None
            complete = True
            for v0, i in scored[:root_beam]:
                if _over(deadline):
                    complete = False
                    break
                action = playable_actions[i]
                try:
                    if action.action_type == ActionType.END_TURN:
                        continue
                    if is_chance(action):
                        # Keep the depth-1 (expected) value: expanding a ROLL into 11 outcomes at
                        # every depth is what made search bots run out of clock, and within a turn
                        # rolling before or after a build gives the same position.
                        continue
                    else:
                        succ = world.copy()
                        succ.execute(action)
                        if succ.winning_color() is not None and succ.winning_color() == me:
                            return i, WIN
                        if succ.state.current_color() != me:
                            val = value_fn(succ, me)
                        else:
                            val = _continuation_value(succ, depth - 1, deadline,
                                                      value_fn, me, beam)
                except Exception:
                    continue
                if val > improved_val:
                    improved_val, improved_idx = val, i
            if not complete:
                break
            if improved_idx is not None and improved_val > best_val:
                best_val, best_idx = improved_val, improved_idx
            # A completed depth did not change the choice: deeper levels rarely will; keep going
            # only while time is left (the loop condition already checks the deadline).

    if not isinstance(best_idx, int) or not 0 <= best_idx < len(playable_actions):
        best_idx = 0
    return best_idx, best_val
