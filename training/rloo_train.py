from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="RLOO fallback training entrypoint placeholder.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-dir", default="outputs/rloo-smoke")
    parser.parse_args()
    raise SystemExit("RLOO fallback is planned after GRPO reward smoke tests.")


if __name__ == "__main__":
    main()

