"""version_6: SUB3 (agent2 + v4 value-net override) with the clock made safe, the net consulted on
every main turn, a wider set of decisions the net may override, value-function weight variants, a
win finder, and an opening model hook.

Everything is behind the SWITCHES block, so each idea can be A/B tested on its own.

    a2.py         agent2: rules + planner + counter-opening      (working copy of SUB3's)
    v4net.py      the v4 value-net ensemble                     (working copy of SUB3's)
    feats.py      the value-net features                        (working copy of SUB3's)
    fastcopy.py   copy-on-write game states                     (working copy of SUB3's)
    winfind.py    exact-ish win search for the last turn
    open_logreg.json  the learned opening evaluator

What changed relative to `version_6/base` (SUB3).  Each item is one switch:

  open_cap         SUB3's counter-opening search has no deadline.  Measured: one first decision of
                   233 ms on this machine against a 312 ms first allowance, and 2 games in 800 ran
                   out of clock (23 randomly played moves).  open_cap caps what that search may
                   spend, as a share of the allowance that is actually left.
  reserve/cap_frac agent2's own search budget, as instance attributes (was a module constant).
  net_budget       SUB3 gated the net on a fixed NET_MIN_CPU = 0.25 s, so it was skipped in 3% of
                   games.  "adaptive" gates on the *measured* cost of one net call instead.
  net_lookahead    the net's own depth-2 continuation can be switched off.
  net_on_robber / net_on_knight / net_on_discard   let the net override those rules too, under the
                   same margin rule.
  margin           P(win) the net must gain over agent2's pick before it overrides it.
  vf               value-function weight overrides for agent2 (a dict merged over a2.WEIGHTS).
  winfind_vp       run winfind (depth winfind_depth, winfind_budget copies) from this actual VP
                   total, before anything else.

`decide()` never raises.  `self.stats` counts main / net / override; the lab reads it, the agent
never prints.
"""
import os

# The harness runs one agent per process on one core with OMP/MKL pinned to 1.  Pinning OpenBLAS too
# costs nothing and keeps the lab's in-process timing close to the harness.  (Measured after this:
# no effect on the harness's CPU accounting, because numpy is imported before the agent.)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import random          # noqa: E402
import sys             # noqa: E402
import time            # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import a2              # noqa: E402
import v4net           # noqa: E402

# SWITCHES-BEGIN
SWITCHES = {
    "margin": 0.03,
    "net_budget": "adaptive",     # "fixed" (SUB3) | "adaptive" | "always" | "off"
    "net_reserve": 0.10,          # seconds of allowance kept back
    "net_costs": 3.0,             # ... plus this many measured net calls
    "net_lookahead": True,
    "net_on_robber": False,
    "net_on_knight": False,
    "net_on_discard": False,
    "open_cap": 0.24,             # seconds the opening search may spend; 99 = SUB3 (no deadline)
    "opening": "counter",         # "counter" (SUB3, the learned counter-opening) | "ply" (1-ply)
    "reserve": 0.10,              # agent2 planner: seconds of allowance never spent
    "cap_frac": 1.0 / 6.0,        # agent2 planner: max share of the spare allowance per decision
    "winfind_vp": 99,             # actual VP at which the win finder runs (99 = off)
    "winfind_depth": 3,
    "winfind_budget": 400,
    "winfind_min_cpu": 0.12,      # ... and only while this much of the allowance is left
    "vf": {},                     # value-function weight overrides over a2.WEIGHTS
}
# SWITCHES-END

NET_MIN_CPU = 0.25        # SUB3's fixed threshold, used when net_budget == "fixed"


class Agent(a2.Agent):
    def __init__(self, switches=None):
        cfg = dict(SWITCHES)
        if isinstance(switches, dict):
            cfg.update(switches)
        self.cfg = cfg
        super().__init__()                       # builds self.rng and the plain vf
        # per-instance settings (never module globals: two agents can share one process in the lab)
        self.vf = a2.make_vf({**a2.WEIGHTS, **(cfg.get("vf") or {})})
        self.open_cap = float(cfg["open_cap"])
        self.opening = cfg["opening"]
        self.reserve = float(cfg["reserve"])
        self.cap_frac = float(cfg["cap_frac"])
        self.net = v4net.Agent(lookahead=bool(cfg["net_lookahead"]))
        self._obs = None
        self.stats = dict(main=0, net=0, override=0, winfind=0, winhit=0)
        self.net_cost = 0.0025       # EWMA of the measured CPU of one net.scores call, seconds
        self._winfind = None
        if float(cfg["winfind_vp"]) < 99:
            try:
                import winfind
                self._winfind = winfind
            except Exception:
                self._winfind = None

    # ------------------------------------------------------------------ entry point
    def decide(self, obs, actions):
        self._obs = obs
        try:
            if self._winfind is not None and getattr(obs, "my_actual_vps", 0) >= self.cfg["winfind_vp"]:
                self.stats["winfind"] += 1
                i = self._try_win(obs, actions)
                if i is not None:
                    self.stats["winhit"] += 1
                    return i
        except Exception:
            pass
        return super().decide(obs, actions)

    def _try_win(self, obs, actions):
        try:
            if len(actions) <= 1:
                return 0
            cpu = getattr(obs, "cpu_left", None)
            if cpu is not None and cpu < float(self.cfg["winfind_min_cpu"]):
                return None
            world = obs.sample_world(self.rng)
            a = self._winfind.find_win(world, obs.me, depth=int(self.cfg["winfind_depth"]),
                                       budget=int(self.cfg["winfind_budget"]))
            if a is None:
                return None
            for i, b in enumerate(actions):
                if b == a:
                    return i
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ net budgeting
    def _net_ok(self):
        obs = self._obs
        cpu = getattr(obs, "cpu_left", None) if obs is not None else None
        if cpu is None:
            return True
        mode = self.cfg["net_budget"]
        if mode == "off":
            return False
        if mode == "always":
            return True
        if mode == "fixed":
            return cpu >= NET_MIN_CPU
        return cpu >= float(self.cfg["net_reserve"]) + float(self.cfg["net_costs"]) * self.net_cost

    def _net_pick(self, actions, world=None):
        """The net's best index in `actions`, or None.  Updates the cost EWMA."""
        if self._obs is None or not self._net_ok():
            return None
        t0 = time.process_time()
        try:
            vals = self.net.scores(self._obs, actions, world=world)
        except Exception:
            return None
        dt = time.process_time() - t0
        self.net_cost = 0.8 * self.net_cost + 0.2 * dt
        self.stats["net"] += 1
        try:
            return int(vals.argmax()), vals
        except Exception:
            return None

    def _maybe_override(self, i, actions, world=None):
        got = self._net_pick(actions, world=world)
        if got is None:
            return i
        j, vals = got
        try:
            if j != i and vals[j] - vals[i] > float(self.cfg["margin"]):
                self.stats["override"] += 1
                return j
        except Exception:
            pass
        return i

    # ------------------------------------------------------------------ main turn
    def _main(self, world, me, actions, deadline):
        i = super()._main(world, me, actions, deadline)
        self.stats["main"] += 1
        if len(actions) <= 1:
            return i
        return self._maybe_override(i, actions, world=world)

    # ------------------------------------------------------------------ other decisions
    def _robber(self, st, me, opp, actions):
        i = super()._robber(st, me, opp, actions)
        if self.cfg["net_on_robber"] and len(actions) > 1:
            return self._maybe_override(i, actions)
        return i

    def _pre_roll(self, st, me, opp, actions):
        i = super()._pre_roll(st, me, opp, actions)
        if self.cfg["net_on_knight"] and len(actions) > 1:
            return self._maybe_override(i, actions)
        return i

    def _discard(self, st, me, actions):
        i = super()._discard(st, me, actions)
        if self.cfg["net_on_discard"] and len(actions) > 1:
            return self._maybe_override(i, actions)
        return i
