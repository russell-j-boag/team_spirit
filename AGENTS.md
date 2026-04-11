# RL Agent Action Plan

This repo has five comparison lanes (`P2`, `P3`, `P4`, `P5`, `P6`).
All agent families are exposed as selectable variants and should be rotated into the active lineup for focused A/B/C comparisons.

## Current architecture

- Lightweight config layer is implemented in `python/ts.py`:
- `BOT_BASE_CONFIGS`: per-lane defaults and hyperparameters.
- `AGENT_VARIANT_OVERRIDES`: one unique entry per algorithm variant.
- `ACTIVE_AGENT_VARIANTS`: current lineup selector.
- `build_agent_controller(owner)`: instantiates lane controllers from config (no manual constructor rewiring in `main()`).
- Runtime and smoke runs should use the `r-pygame` conda env (for example `conda run --no-capture-output -n r-pygame python -u python/ts.py`).

## Implemented variants (no duplicates)

Tabular control variants:

1. `Q-learning`
2. `SARSA`
3. `SARSA(lambda)`
4. `Watkins Q(lambda)`
5. `Double Q-learning`
6. `Expected SARSA`
7. `N-step SARSA`
8. `Differential SARSA`
9. `R-learning`
10. `Dyna-Q`
11. `UCB Q-learning`
12. `Optimistic-initialization Q-learning`
13. `Hysteretic Q-learning`
14. `Lenient Q-learning`

Bayesian control variant:

1. `Bayesian Q` (Thompson sampling option)

## Current active lineup

1. `P2`: `SARSA_LAMBDA`
2. `P3`: `WATKINS_Q_LAMBDA`
3. `P4`: `BAYES_Q`
4. `P5`: `DIFFERENTIAL_SARSA`
5. `P6`: `DYNA_Q`

## Operating rule for swaps

1. Keep only five active lanes at once.
2. Change `ACTIVE_AGENT_VARIANTS` only; do not edit constructor wiring.
3. Keep `AGENT_VARIANT_OVERRIDES` as the single source of variant definitions.
4. Do not add duplicate entries pointing to the same algorithm constant.

## Next focus

1. Run systematic comparisons for the trace pair vs. tabular and Bayesian baselines.
2. Benchmark average-reward (`Differential SARSA`, `R-learning`) against discounted control.
3. Benchmark planning/exploration variants (`Dyna-Q`, `UCB`, `Optimistic`) under identical seeds/settings.
4. Benchmark non-stationary-friendly variants (`Hysteretic`, `Lenient`) in the shared environment.
