from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO entrypoint for AlgoAgent preference data.")
    parser.add_argument("--model", default="outputs/sft-qwen25-coder")
    parser.add_argument("--dataset", default="data/processed/dpo.jsonl")
    parser.add_argument("--output-dir", default="outputs/dpo-qwen25-coder")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install them on the GPU server with "
            "`pip install -r requirements-train.txt`."
        ) from exc

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", trust_remote_code=True)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        num_train_epochs=1,
        beta=0.1,
        logging_steps=10,
        save_steps=200,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

