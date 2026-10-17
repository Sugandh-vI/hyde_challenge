"""Same-turn search: F's depth-1 pass, then optional iterative deepening.

Design (unchanged from what we learned in my_team/, see version_3/EXPERIMENTS.md):

* The depth-1 pass is F's rule: execute every legal action on one sampled world, score the
  successor with the value function, take the max.  It is ALWAYS completed.  A truncated depth-1
  pass returns the best of an arbitrary prefix of the action list and is much worse than F
  (measured: 0.125 against F when it was cut short).
* Deeper passes deepen only the best ``root_beam`` root actions.  Inside a line we keep the
  ``beam`` best children, stop at END_TURN / a change of turn, and never recurse through a chance
  event (we use its probability-weighted value instead).  Rolling before or after a build reaches
  the same position within a turn, so this is cheap and loses almost nothing.
* Chance nodes are expanded with the engine's ``execute_spectrum``.  Robber steals get their true
  probabilities: in 2-player Catan the victim's hand is public (19 - bank - mine), and the engine
  otherwise assigns 1/5 to each resource.
* No pruning from ``list_prunned_actions``: its robber pruning (a single "most impactful" tile)
  cost us ~10 points of win rate in my_team/ (EXPERIMENTS.md E3/E4).
"""

import time

from catanatron.models.enums import ActionType
from catanatron.players.tree_search_utils import execute_spectrum
from catanatron.state_functions import get_player_freqdeck

CHANCE_TYPES = frozenset([ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD])
RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
WIN = float("inf")


def _is_chance(action):
    at = action.action_type
    if at in CHANCE_TYPES:
        return True
    if at == ActionType.MOVE_ROBBER:
        try:
            return action.value[1] is not None
        except Exception:
            return False
    return False


def _chance_outcomes(game, action):
    try:
        outcomes = execute_spectrum(game, action)
    except Exception:
        return None
    if not outcomes:
        return None
    if action.action_type == ActionType.MOVE_ROBBER:
        try:
            victim = action.value[1]
            hand = get_player_freqdeck(game.state, victim)
            total = float(sum(hand))
            if total > 0:
                fixed = []
                for g, _p in outcomes:
                    card = g.state.action_records[-1].result
                    if card in RESOURCES:
                        fixed.append((g, hand[RESOURCES.index(card)] / total))
                if fixed and sum(p for _g, p in fixed) > 0:
                    tot = sum(p for _g, p in fixed)
                    return [(g, p / tot) for g, p in fixed]
        except Exception:
            pass
    return outcomes


def _chance_value(outcomes, vf, me):
    total, weight = 0.0, 0.0
    for g, p in outcomes:
        try:
            if g.winning_color() == me:
                return WIN, True
            total += p * vf(g, me)
            weight += p
        except Exception:
            continue
    if weight <= 0:
        return None, False
    return total / weight, False


def eval_action(game, action, vf, me):
    """(value, successor_or_None, won).  Chance actions are probability-weighted."""
    if _is_chance(action):
        outcomes = _chance_outcomes(game, action)
        if outcomes is not None:
            v, won = _chance_value(outcomes, vf, me)
            if won:
                return WIN, None, True
            if v is not None:
                return v, None, False
    succ = game.copy()
    succ.execute(action)
    if succ.winning_color() is not None and succ.winning_color() == me:
        return WIN, succ, True
    return vf(succ, me), succ, False


def _continuation(game, depth, deadline, vf, me, beam):
    try:
        if depth <= 0 or time.process_time() >= deadline:
            return vf(game, me)
        w = game.winning_color()
        if w is not None:
            return WIN if w == me else vf(game, me)
        if game.state.current_color() != me:
            return vf(game, me)
    except Exception:
        try:
            return vf(game, me)
        except Exception:
            return 0.0

    base = vf(game, me)                      # value of stopping here (about END_TURN)
    scored = []
    for a in list(game.playable_actions):
        if time.process_time() >= deadline:
            return base
        try:
            v, succ, won = eval_action(game, a, vf, me)
        except Exception:
            continue
        if won:
            return WIN
        scored.append((v, a, succ))
    if not scored:
        return base
    scored.sort(key=lambda x: x[0], reverse=True)
    best = base
    for v, a, succ in scored[:beam]:
        if a.action_type == ActionType.END_TURN or succ is None:
            cand = v
        else:
            try:
                if succ.state.current_color() != me or succ.winning_color() is not None:
                    cand = v
                else:
                    cand = _continuation(succ, depth - 1, deadline, vf, me, beam)
            except Exception:
                cand = v
        if cand > best:
            best = cand
    return best


def choose_action(world, me, playable_actions, vf, budget_s,
                  beam=3, root_beam=5, max_depth=3, safety=0.8):
    """Return (index, value) of the best action in ``playable_actions``."""
    if not playable_actions:
        return 0, 0.0
    start = time.process_time()
    deadline = start + max(0.0, budget_s) * safety

    # ---- depth 1: F's pass, always completed --------------------------------------------
    scored = []
    for i, a in enumerate(playable_actions):
        try:
            v, _succ, won = eval_action(world, a, vf, me)
        except Exception:
            continue
        if won:
            return i, WIN
        scored.append((v, i))
    if not scored:
        return 0, 0.0
    scored.sort(key=lambda x: x[0], reverse=True)
    best_v, best_i = scored[0]

    # ---- deeper passes, deadline-limited --------------------------------------------------
    if max_depth > 1 and budget_s > 0.002:
        for depth in range(2, max_depth + 1):
            if time.process_time() >= deadline:
                break
            improved_v, improved_i = -float("inf"), None
            complete = True
            for _v0, i in scored[:root_beam]:
                if time.process_time() >= deadline:
                    complete = False
                    break
                a = playable_actions[i]
                if a.action_type == ActionType.END_TURN or _is_chance(a):
                    continue
                try:
                    succ = world.copy()
                    succ.execute(a)
                    if succ.winning_color() is not None and succ.winning_color() == me:
                        return i, WIN
                    if succ.state.current_color() != me:
                        v = vf(succ, me)
                    else:
                        v = _continuation(succ, depth - 1, deadline, vf, me, beam)
                except Exception:
                    continue
                if v > improved_v:
                    improved_v, improved_i = v, i
            if not complete:
                break
            if improved_i is not None and improved_v > best_v:
                best_v, best_i = improved_v, improved_i

    if not isinstance(best_i, int) or not 0 <= best_i < len(playable_actions):
        best_i = 0
    return best_i, best_v
