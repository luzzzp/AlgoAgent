from __future__ import annotations

import re

from algoagent.model_client import ModelResponse
from algoagent.schema import ComplexityEstimate, ProblemSpec


class HuggingFaceModel:
    """Optional local Transformers backend for base or post-trained models."""

    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: str = "",
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
        load_in_4bit: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace backend requires torch and transformers. "
                "Install GPU dependencies with `pip install -r requirements-train.txt`."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError("4-bit loading requires bitsandbytes and a recent transformers build.") from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Loading a LoRA adapter requires peft. Install requirements-train.txt.") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model.eval()

    def generate_solution(
        self,
        problem: ProblemSpec,
        feedback: str | None,
        attempt: int,
    ) -> ModelResponse:
        prompt = problem.prompt()
        prompt += (
            f"\nAttempt: {attempt}\n"
            "Use this exact response structure:\n"
            "Time Complexity: O(...)\nSpace Complexity: O(...)\n```cpp\n...\n```\n"
        )
        if feedback:
            prompt += f"\nPrevious attempt feedback:\n{feedback[:4000]}\n"
        completion = self._generate(
            "You are AlgoAgent. Independently choose an algorithm that satisfies the stated limits.",
            prompt,
        )
        return ModelResponse(
            draft_reasoning=completion[:1000],
            code=_extract_code(completion),
            complexity=ComplexityEstimate(
                time_complexity=_extract_complexity(completion, "time"),
                space_complexity=_extract_complexity(completion, "space"),
            ),
        )

    def explain_solution(
        self,
        problem: ProblemSpec,
        verified_code: str,
        complexity: ComplexityEstimate,
    ) -> str:
        return self._generate(
            "Explain the verified algorithm concisely in Chinese. Do not change the code.",
            (
                f"{problem.prompt()}\n\nVerified code:\n```cpp\n{verified_code}\n```\n"
                f"Time complexity: {complexity.time_complexity}\n"
                f"Space complexity: {complexity.space_complexity}\n"
            ),
        ).strip()

    def _generate(self, system: str, prompt: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generate_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
        with self.torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)
        generated = output_ids[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:cpp|c\+\+)?\s*(.*?)```", text, re.S | re.I)
    if match:
        return match.group(1).strip()
    include_at = text.find("#include")
    return text[include_at:].strip() if include_at >= 0 else text.strip()


def _extract_complexity(text: str, kind: str) -> str:
    pattern = rf"{kind}\s+complexity\s*:\s*(o\s*\([^)]+\))"
    match = re.search(pattern, text, re.I)
    return match.group(1) if match else "unknown"
