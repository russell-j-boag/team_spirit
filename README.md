# Team Spirit

`team_spirit` is a small `pygame` air-combat simulation with one human player and three learning agents.

The task is to survive waves of incoming enemies while comparing different control policies in the same environment:
- `P1` is the human-controlled jet.
- `P2` and `P3` are tabular RL agents.
- `P4` is a Bayesian Q-learning agent.

The game tracks score, kills, misses, accuracy, targeting behavior, and per-agent Q heatmaps during play. Q-tables are loaded from and saved to the [`json/`](./json) directory.

## Run

From R:

```r
source("1-run.R")
```

This uses `reticulate` and the `r-pygame` conda environment to launch [`python/ts.py`](./python/ts.py).

Directly from Python:

```bash
python python/ts.py
```

## Controls

- Arrow keys: move `P1`
- Space: fire
- `M`: toggle `TEAM` / `VERSUS`
- `T`: toggle training mode for bots
- `Esc`: quit

## Files

- [`python/ts.py`](./python/ts.py): main simulation
- [`python/images/`](./python/images): sprites and backgrounds
- [`json/`](./json): saved Q-table JSON files
- [`1-run.R`](./1-run.R): R launcher
