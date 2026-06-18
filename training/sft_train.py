from __future__ import annotations

import argparse
import inspect

from algoagent.hf_model import SFT_SYSTEM_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT for AlgoAgent data.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--dataset", default="data/processed/solver_sft.jsonl")
    parser.add_argument("--output-dir", default="outputs/solver-sft")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit("Install requirements-train.txt before training.") from exc

    dataset = load_dataset("json", data_files=args.dataset, split="train").map(lambda row: {"text": _format(row)})
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4"),
        device_map="auto",
        trust_remote_code=True,
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        **_trainer_kwargs(
            SFTTrainer,
            model,
            _sft_config(SFTConfig, args.output_dir, args.max_seq_length),
            dataset,
            tokenizer,
            peft_config,
        )
    )
    trainer.train()
    trainer.save_model(args.output_dir)


def _format(row: dict[str, str]) -> str:
    return (
        f"<|im_start|>system\n{SFT_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{row['instruction']}\n{row['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{row['output']}<|im_end|>"
    )


def _sft_config(SFTConfig, output_dir: str, max_seq_length: int):
    signature = inspect.signature(SFTConfig)
    kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "num_train_epochs": 2,
        "logging_steps": 10,
        "save_steps": 200,
    }
    if "max_seq_length" in signature.parameters:
        kwargs["max_seq_length"] = max_seq_length
    elif "max_length" in signature.parameters:
        kwargs["max_length"] = max_seq_length
    if "dataset_text_field" in signature.parameters:
        kwargs["dataset_text_field"] = "text"
    return SFTConfig(**kwargs)


def _trainer_kwargs(SFTTrainer, model, args, dataset, tokenizer, peft_config):
    signature = inspect.signature(SFTTrainer.__init__)
    kwargs = {"model": model, "args": args, "train_dataset": dataset, "peft_config": peft_config}
    if "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    return kwargs


if __name__ == "__main__":
    main()
