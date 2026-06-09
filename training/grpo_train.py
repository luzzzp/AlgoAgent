from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Small GRPO experiment entrypoint for AlgoAgent.")
    parser.add_argument("--model", default="outputs/sft-qwen25-coder")
    parser.add_argument("--dataset", default="data/processed/grpo_prompts.jsonl")
    parser.add_argument("--output-dir", default="outputs/grpo-qwen25-coder")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install them on the GPU server with "
            "`pip install -r requirements-train.txt`."
        ) from exc

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", trust_remote_code=True)
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        num_train_epochs=1,
        num_generations=4,
        logging_steps=10,
    )
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_funcs=[format_reward],
    )
    trainer.train()
    trainer.save_model(args.output_dir)


def format_reward(completions: list[str], **_: object) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        score = 0.0
        if "```cpp" in completion:
            score += 0.2
        if "Complexity:" in completion:
            score += 0.2
        if "#include" in completion and "int main" in completion:
            score += 0.2
        rewards.append(score)
    return rewards


if __name__ == "__main__":
    main()

