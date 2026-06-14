from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training entrypoint placeholder for AlgoAgent-Py.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--prompts", required=True, help="JSONL prompts built from problem statements.")
    parser.add_argument("--output-dir", default="outputs/grpo-smoke")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.parse_args()
    raise SystemExit(
        "GRPO training requires online generation + reward execution on the GPU server. "
        "This entrypoint is reserved; implement after Solver SFT and Python reward smoke pass."
    )


if __name__ == "__main__":
    main()

