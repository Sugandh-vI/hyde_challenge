# preset `s4_net`

Step 4. s2_clock with the net allowed to override the pre-roll knight rule (the only one of the three widened decisions that did not lose: -0.1 pt +-1.7).

Self-contained: `agent.py` is the entry point the kit loads, `core.py` is
`version_6/v6/agent.py`, and every helper module, the value-net weights and the opening model sit in
this folder.  Nothing is imported from outside it.

```bash
python -m catan_kit.evaluate --a version_6/presets/s4_net/agent.py --b examples/f_agent.py --games 1000
```

The switch values, the 800-game paired screens behind them and the final harness numbers are in
`version_6/EXPERIMENTS.md` (and in the README's *Arena update 5*).
