"""Find a sequence of up to `depth` of my own deterministic actions that wins this turn."""
from catanatron.models.enums import ActionType

SKIP = {ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD, ActionType.MOVE_ROBBER,
        ActionType.END_TURN, ActionType.DISCARD_RESOURCE, ActionType.PLAY_KNIGHT_CARD}


def find_win(game, color, depth=3, budget=400):
    """Return the first action of a winning sequence, or None. `budget` caps game copies."""
    count = [0]

    def rec(g, d):
        for a in g.playable_actions:
            if a.action_type in SKIP or count[0] >= budget:
                continue
            count[0] += 1
            g2 = g.copy()
            try:
                g2.execute(a)
            except Exception:
                continue
            if g2.winning_color() == color:
                return True
            if d > 1 and g2.state.current_color() == color and rec(g2, d - 1):
                return True
        return False

    for a in game.playable_actions:
        if a.action_type in SKIP:
            continue
        count[0] += 1
        g2 = game.copy()
        try:
            g2.execute(a)
        except Exception:
            continue
        if g2.winning_color() == color or (depth > 1 and g2.state.current_color() == color
                                           and rec(g2, depth - 1)):
            return a
    return None
