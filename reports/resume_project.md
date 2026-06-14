# AlgoAgent-Py Resume Draft

**AlgoAgent-Py: Verifiable Python Algorithm Problem Agent**

- Built a Python-first algorithm problem agent that generates Chinese solution explanations, complexity analysis, and Python 3 code from programming problem statements.
- Designed a no-leakage evaluation pipeline separating visible tests, reward tests, and held-out eval tests.
- Implemented subprocess-based Python execution with per-test timeout, runtime error capture, output normalization, and output length guards.
- Planned executable reward training with GRPO/RLOO, where code correctness dominates reward and explanation quality is gated by test pass rate.
- Added follow-up dialogue design so users can ask about verified code lines, complexity, edge cases, and algorithm choices.

