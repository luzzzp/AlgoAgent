# AlgoAgent-Py Project Log

## 2026-06-14: Python-first restart

Goal:

- Restart from a clean Python-first codebase.
- Avoid Python-to-C++ translation cost.
- Use verified Python oracle solutions to build larger Solver SFT data.

Key decisions:

- Default output language is Python 3.
- Solver SFT and follow-up Dialogue SFT are separate datasets.
- Reward training should use executable tests, with explanation reward gated by code correctness.
- Agent validation must enforce per-test time limits to avoid dead loops and resource waste.

Server choice:

- Prefer A800 80GB for formal training and GRPO/RLOO smoke.
- A100 40GB is acceptable for cheap 7B SFT smoke but less comfortable for RL and long contexts.

