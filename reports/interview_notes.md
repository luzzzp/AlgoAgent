# Interview Notes

## Why Python-first?

The previous C++ route required Python-to-C++ translation and strict re-verification, which made data production slow. The Python-first route directly uses verified Python oracle solutions, enabling larger SFT data and faster RL/Agent iteration.

## Why not multi-agent?

Algorithm problem solving has a stable workflow: solve, execute, repair, verify, and explain. A single state-machine Agent is easier to control, easier to evaluate, and avoids unnecessary coordination complexity.

## Why GRPO/RLOO?

Algorithm tasks have executable rewards. GRPO/RLOO can sample multiple solutions per problem and optimize against test pass rates, syntax validity, timeout penalties, and gated explanation quality.

