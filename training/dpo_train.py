from __future__ import annotations

import argparse
import inspect


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO entrypoint for AlgoAgent preference data.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter", default="", help="Optional SFT LoRA adapter to continue training from.")
    parser.add_argument("--dataset", default="data/processed/dpo.jsonl")
    parser.add_argument("--output-dir", default="outputs/dpo-qwen25-coder")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA loading.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install them on the GPU server with "
            "`pip install -r requirements-train.txt`."
        ) from exc

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if not args.no_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    peft_config = None
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
    training_args = _build_dpo_config(
        DPOConfig,
        output_dir=args.output_dir,
    )
    trainer = DPOTrainer(
        **_build_dpo_trainer_kwargs(
            DPOTrainer,
            model=model,
            training_args=training_args,
            dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
    )
    trainer.train()
    trainer.save_model(args.output_dir)


def _build_dpo_config(DPOConfig, output_dir: str):
    signature = inspect.signature(DPOConfig)
    kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-6,
        "num_train_epochs": 1,
        "logging_steps": 10,
        "save_steps": 200,
    }
    if "beta" in signature.parameters:
        kwargs["beta"] = 0.1
    return DPOConfig(**kwargs)


def _build_dpo_trainer_kwargs(
    DPOTrainer,
    *,
    model,
    training_args,
    dataset,
    tokenizer,
    peft_config,
) -> dict[str, object]:
    signature = inspect.signature(DPOTrainer.__init__)
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
    }
    if peft_config is not None and "peft_config" in signature.parameters:
        kwargs["peft_config"] = peft_config
    if "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    return kwargs


if __name__ == "__main__":
    main()
