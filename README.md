# AlgoAgent-Py

AlgoAgent-Py is a Python-first algorithm problem agent project. It trains a solver to output Chinese solution explanations, complexity analysis, and Python 3 code, then verifies and repairs generated programs with executable tests.

## Project Stages

1. Convert TACO-verified Python problems into AlgoAgent-Py JSON.
2. Verify Python oracle solutions by executing visible, reward, and eval tests.
3. Build Solver SFT data.
4. Build follow-up SFT data from verified solutions.
5. Merge Solver + follow-up SFT data and train the SFT model.
6. Evaluate format, follow-up behavior, and executable correctness.
7. Run Agent repair with visible-test feedback.
8. Add GRPO/RLOO reward training smoke experiments after SFT is complete.

## Quick Smoke Test

```bash
python -m unittest discover -s tests -v
```

## Data Commands

Solver SFT uses one record per verified problem. The full problem statement is stored in the `input`
field, while the `output` field contains the Chinese explanation, complexity, and verified Python code.
Follow-up SFT uses multiple records per verified problem and asks an LLM to generate diverse student-style
questions grounded in the statement and code. Each generated follow-up item must choose an `evidence_id`
from provided statement/code/annotation snippets; code evidence ids use real line numbers such as `C25`.
Unsupported or hidden-test-leaking items are filtered out.

```bash
python scripts/convert_taco_python.py \
  --dataset likaixin/TACO-verified \
  --split train \
  --out-dir data/problems/taco_python_1000 \
  --limit 1000

python scripts/verify_python_oracles.py \
  --problems data/problems/taco_python_1000 \
  --out-dir data/problems/taco_python_1000_verified

python scripts/generate_solution_annotations.py \
  --problems data/problems/taco_python_1000_verified \
  --out data/processed/taco_python_1000_annotations.jsonl \
  --backend hf \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --load-in-4bit \
  --resume

python scripts/make_solver_sft.py \
  --problems data/problems/taco_python_1000_verified \
  --annotations data/processed/taco_python_1000_annotations.jsonl \
  --out-dir data/processed/taco_python_1000_solver_sft

python scripts/make_dialogue_sft.py \
  --problems data/problems/taco_python_1000_verified \
  --annotations data/processed/taco_python_1000_annotations.jsonl \
  --out-dir data/processed/taco_python_1000_followup_sft \
  --backend hf \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --load-in-4bit \
  --max-questions-per-problem 5 \
  --resume

python scripts/merge_sft_datasets.py \
  --inputs \
    data/processed/taco_python_1000_solver_sft/solver_sft.jsonl \
    data/processed/taco_python_1000_followup_sft/followup_sft.jsonl \
  --out data/processed/taco_python_1000_sft/combined_sft.jsonl
```

## Training

```bash
python training/sft_train.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset data/processed/taco_python_1000_sft/combined_sft.jsonl \
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
