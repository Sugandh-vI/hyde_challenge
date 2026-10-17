# preset `s2_clock`

Step 2. version_6 defaults: the value net on every main turn with an adaptive CPU budget instead of SUB3's fixed NET_MIN_CPU = 0.25 s, the counter-opening search capped at 0.24 s, agent2's planner budget as instance settings.

Self-contained: `agent.py` is the entry point the kit loads, `core.py` is
`version_6/v6/agent.py`, and every helper module, the value-net weights and the opening model sit in
this folder.  Nothing is imported from outside it.

```bash
python -m catan_kit.evaluate --a version_6/presets/s2_clock/agent.py --b examples/f_agent.py --games 1000
```

The switch values, the 800-game paired screens behind them and the final harness numbers are in
`version_6/EXPERIMENTS.md` (and in the README's *Arena update 5*).
