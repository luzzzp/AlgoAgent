# AlgoAgent-Py

AlgoAgent-Py is a Python-first algorithm problem agent project. It trains a solver to output Chinese solution explanations, complexity analysis, and Python 3 code, then verifies and repairs generated programs with executable tests.

## Project Stages

1. Convert TACO-verified Python problems into AlgoAgent-Py JSON.
2. Verify Python oracle solutions by executing visible, reward, and eval tests.
3. Build Solver SFT data.
4. Train Solver SFT.
5. Evaluate format and executable correctness.
6. Run Agent repair with visible-test feedback.
7. Add GRPO/RLOO reward training smoke experiments.
8. Build follow-up dialogue data from verified solutions.

## Quick Smoke Test

```bash
python -m unittest discover -s tests -v
```

## Data Commands

```bash
python scripts/convert_taco_python.py \
  --dataset likaixin/TACO-verified \
  --split train \
  --out-dir data/problems/taco_python_1000 \
  --limit 1000

python scripts/verify_python_oracles.py \
  --problems data/problems/taco_python_1000 \
  --out-dir data/problems/taco_python_1000_verified

python scripts/make_solver_sft.py \
  --problems data/problems/taco_python_1000_verified \
  --out-dir data/processed/taco_python_1000_solver_sft
```

## Training

```bash
python training/sft_train.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset data/processed/taco_python_1000_solver_sft/solver_sft.jsonl \
  --output-dir outputs/solver-sft-qwen25-coder
```

## Evaluation

```bash
python scripts/evaluate_format.py \
  --problems data/problems/taco_python_1000_verified \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter outputs/solver-sft-qwen25-coder \
  --out reports/solver_sft_format.json \
  --limit 500 \
  --resume
```

## Server Recommendation

Use A800 80GB for the formal run. A100 40GB is cheaper but more likely to hit memory limits during GRPO/RLOO and multi-sample generation.

