# preset `s5_win`

Step 5. s2_clock plus the depth-3 win finder from actual VP >= 8 (+0.3 pt +-0.3, but 3 of 800 games ran out of clock: NOT recommended).

Self-contained: `agent.py` is the entry point the kit loads, `core.py` is
`version_6/v6/agent.py`, and every helper module, the value-net weights and the opening model sit in
this folder.  Nothing is imported from outside it.

```bash
python -m catan_kit.evaluate --a version_6/presets/s5_win/agent.py --b examples/f_agent.py --games 1000
```

The switch values, the 800-game paired screens behind them and the final harness numbers are in
`version_6/EXPERIMENTS.md` (and in the README's *Arena update 5*).
