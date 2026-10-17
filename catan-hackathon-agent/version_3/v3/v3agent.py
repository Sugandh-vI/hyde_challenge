"""version_3 agent: counter-opening + F-style 1-ply play + our extra value terms.

Everything is behind a preset (``PRESET`` below, or ``Agent(preset="P4")``):

    P1  opening(f_value) + value extras, no rules, no search
    P2  P1 + RULES (robber / discard / endgame)
    P3  P2 + SEARCH (same-turn deepening, depth 1 always completed)
    P4  P3 with OPP_MODEL = "mix"     (opening robust to non-F opponents)
    P5  P3 with OPP_MODEL = "generic" (opening ignores F entirely)

Fallback chain for every decision: full policy -> plain 1-ply with F's own value function ->
first ROLL/END_TURN -> index 0.  ``decide()`` never raises.

CPU: the budget per decision comes from ``obs.cpu_left`` (300 ms at the start plus 12.5 ms per
decision, capped at 1.5 s by the harness).  The opening spends the largest share because it
matters most and the clock is largest relative to the work there; the mid-game caps are small.
"""

import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)          # siblings are imported by plain module name

from catanatron.models.enums import CITY, SETTLEMENT, ActionType
from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn

import v3_opening
import v3_rules
import v3_search
import v3_value

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS = {
    # P0 is a diagnostic only (not one of the five presets): F + value extras, no opening
    # search, no rules, no search.  It isolates the opening's contribution.
    "P0": dict(opening=False, opp_model="f_value", rules=False, search=False),
    "P1": dict(opening=True, opp_model="f_value", rules=False, search=False),
    "P2": dict(opening=True, opp_model="f_value", rules=True, search=False),
    "P3": dict(opening=True, opp_model="f_value", rules=True, search=True),
    "P4": dict(opening=True, opp_model="mix", rules=True, search=True),
    "P5": dict(opening=True, opp_model="generic", rules=True, search=True),
    # P6/P7 are diagnostics for the 2x2 component study (see EXPERIMENTS.md E3).
    "P6": dict(opening=True, opp_model="f_value", rules=False, search=True),
    "P7": dict(opening=True, opp_model="f_value", rules=True, search=False),
}
PRESET = "P3"

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
RESERVE_S = 0.14            # CPU kept for the rest of the game
EST_DECISIONS_LEFT = 38
CAP_NORMAL_S = 0.011        # per-decision cap in the mid-game
OPEN_BUDGET_1_S = 0.16      # our first settlement
OPEN_BUDGET_2_S = 0.08      # our second settlement (a reply is already on the board)
OPEN_BUDGET_ROAD_S = 0.012
OPEN_FRACTION = 0.60        # never spend more than this share of what is left on the opening
CLOCK_GUARD_S = 0.003       # below this, end the turn instead of thinking
SEARCH_DEPTH = 3
SEARCH_BEAM = 3
SEARCH_ROOT_BEAM = 5
ENDGAME_OPP_VP = 8
ENDGAME_DENY = 2.5          # multiplier on enemy_production once the opponent is at 8+ VPs
DUMP = os.environ.get("V3_DUMP")   # diagnostics only; unset in scoring


def _roll_or_end(actions):
    for i, a in enumerate(actions):
        if a.action_type in (ActionType.ROLL, ActionType.END_TURN):
            return i
    return 0


class Agent:
    def __init__(self, preset=None):
        self.preset = preset or PRESET
        self.cfg = dict(PRESETS.get(self.preset, PRESETS["P3"]))
        self.rng = random.Random(0)

        conf = v3_value.load_config(_HERE)
        self.base_weights = conf["base"]
        self.extras = conf["extra"]
        self.vf = v3_value.make_value_fn(self.base_weights, self.extras)
        self.deny_vf = None
        try:
            self.base_fn = get_value_fn("base_fn", DEFAULT_WEIGHTS)
        except Exception:
            self.base_fn = self.vf
        try:
            self.opening_cfg = v3_opening.cfg_from(conf["opening"])
        except Exception:
            self.opening_cfg = v3_opening.cfg_from({})
        self._games = 0

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def decide(self, obs, playable_actions) -> int:
        try:
            return self._decide(obs, playable_actions)
        except BaseException:
            try:
                return self._fallback(obs, playable_actions, self.base_fn)
            except BaseException:
                try:
                    return _roll_or_end(playable_actions)
                except BaseException:
                    return 0

    # ------------------------------------------------------------------
    def _fallback(self, obs, playable_actions, vf):
        """Plain F-style 1-ply on one sampled world."""
        try:
            world = obs.sample_world(self.rng)
        except Exception:
            return _roll_or_end(playable_actions)
        best_i, best_v = 0, None
        for i, a in enumerate(playable_actions):
            try:
                succ = world.copy()
                succ.execute(a)
                if succ.winning_color() is not None and succ.winning_color() == obs.me:
                    return i
                v = vf(succ, obs.me)
                if best_v is None or v > best_v:
                    best_i, best_v = i, v
            except Exception:
                continue
        return best_i

    # ------------------------------------------------------------------
    def _endgame_vf(self):
        if self.deny_vf is None:
            w = dict(self.base_weights)
            try:
                w["enemy_production"] = w["enemy_production"] * ENDGAME_DENY
            except Exception:
                pass
            self.deny_vf = v3_value.make_value_fn(w, self.extras)
        return self.deny_vf

    # ------------------------------------------------------------------
    def _budget(self, cpu_left, opening_kind):
        if cpu_left is None:
            return {"first": OPEN_BUDGET_1_S, "second": OPEN_BUDGET_2_S, "road": 0.012}.get(
                opening_kind, CAP_NORMAL_S)
        try:
            cpu_left = float(cpu_left)
        except Exception:
            cpu_left = 1.0
        if opening_kind in ("first", "second", "road"):
            cap = {"first": OPEN_BUDGET_1_S, "second": OPEN_BUDGET_2_S,
                   "road": OPEN_BUDGET_ROAD_S}[opening_kind]
            return max(0.002, min(cap, cpu_left * OPEN_FRACTION))
        avail = max(0.0, cpu_left - RESERVE_S)
        return max(0.0015, min(CAP_NORMAL_S, avail / EST_DECISIONS_LEFT))

    # ------------------------------------------------------------------
    def _decide(self, obs, playable_actions):
        n = len(playable_actions)
        if n == 0:
            return 0
        if n == 1:
            return 0

        cpu_left = getattr(obs, "cpu_left", None)
        if cpu_left is not None:
            try:
                if float(cpu_left) < CLOCK_GUARD_S:
                    return _roll_or_end(playable_actions)
            except Exception:
                pass

        world = obs.sample_world(self.rng)
        st = world.state
        me = obs.me
        my_color = me
        opp_color = None
        try:
            opp_color = next(c for c in st.colors if c != me)
        except Exception:
            pass
        my_b = st.buildings_by_color[me]
        mine = list(my_b[SETTLEMENT]) + list(my_b[CITY])
        theirs = []
        if opp_color is not None:
            ob = st.buildings_by_color[opp_color]
            theirs = list(ob[SETTLEMENT]) + list(ob[CITY])

        # ---- opening ------------------------------------------------------------------
        if self.cfg["opening"] and getattr(st, "is_initial_build_phase", False):
            try:
                first_type = playable_actions[0].action_type
            except Exception:
                first_type = None
            if first_type == ActionType.BUILD_SETTLEMENT:
                kind = "first" if not mine else "second"
                budget = self._budget(cpu_left, kind)
                t0 = time.process_time()
                cands = [a.value for a in playable_actions
                         if a.action_type == ActionType.BUILD_SETTLEMENT]
                node = v3_opening.choose_settlement(
                    world, cands, mine, theirs, me, self.cfg["opp_model"],
                    self.opening_cfg, self.base_fn, deadline=t0 + budget)
                if node is not None:
                    for i, a in enumerate(playable_actions):
                        if a.action_type == ActionType.BUILD_SETTLEMENT and a.value == node:
                            return i
            elif first_type == ActionType.BUILD_ROAD:
                try:
                    budget = self._budget(cpu_left, "road")
                    if cpu_left is None or float(cpu_left) > 3 * budget:
                        i = v3_opening.choose_road(world, playable_actions, mine, theirs,
                                                   self.opening_cfg)
                        if isinstance(i, int) and 0 <= i < n:
                            return i
                except Exception:
                    pass

        # ---- special rules ------------------------------------------------------------
        if self.cfg["rules"]:
            try:
                types = {a.action_type for a in playable_actions}
            except Exception:
                types = set()
            if ActionType.MOVE_ROBBER in types and n > 1:
                try:
                    i = v3_rules.robber_index(
                        world, playable_actions, me,
                        opp_resources=getattr(obs, "opp_resources", None),
                        endgame=(getattr(obs, "opp_public_vps", 0) or 0) >= ENDGAME_OPP_VP)
                    if isinstance(i, int) and 0 <= i < n:
                        return i
                except Exception:
                    pass
            if ActionType.DISCARD_RESOURCE in types and n > 1:
                try:
                    i = v3_rules.discard_index(playable_actions, obs.my_resources, obs.my_devs)
                    if isinstance(i, int) and 0 <= i < n:
                        return i
                except Exception:
                    pass

        # ---- value function for this decision -----------------------------------------
        vf = self.vf
        try:
            if self.cfg["rules"] and (getattr(obs, "opp_public_vps", 0) or 0) >= ENDGAME_OPP_VP:
                vf = self._endgame_vf()
        except Exception:
            pass

        # ---- 1-ply + optional search ---------------------------------------------------
        budget = self._budget(cpu_left, None)
        if cpu_left is not None and float(cpu_left) < 0.006:
            return self._fallback(obs, playable_actions, self.base_fn)
        depth = SEARCH_DEPTH if self.cfg["search"] else 1
        idx, _v = v3_search.choose_action(
            world, me, playable_actions, vf, budget,
            beam=SEARCH_BEAM, root_beam=SEARCH_ROOT_BEAM, max_depth=depth)
        if type(idx) is int and 0 <= idx < n:
            return idx
        return self._fallback(obs, playable_actions, self.base_fn)
