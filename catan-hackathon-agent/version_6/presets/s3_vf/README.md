# preset `s3_vf`

Step 3. s2_clock with buildable_nodes 1000 -> 2000, the largest point estimate of the value-function weight screen (+0.6 pt, CI +-1.3: NOT significant, do not prefer it over s2_clock on strength).

Self-contained: `agent.py` is the entry point the kit loads, `core.py` is
`version_6/v6/agent.py`, and every helper module, the value-net weights and the opening model sit in
this folder.  Nothing is imported from outside it.

```bash
python -m catan_kit.evaluate --a version_6/presets/s3_vf/agent.py --b examples/f_agent.py --games 1000
```

The switch values, the 800-game paired screens behind them and the final harness numbers are in
`version_6/EXPERIMENTS.md` (and in the README's *Arena update 5*).
