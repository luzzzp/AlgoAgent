from __future__ import annotations

import re

from algoagent.model_client import ModelResponse
from algoagent.schema import ComplexityEstimate, ProblemSpec


class HuggingFaceModel:
    """Optional local Transformers backend for base or post-trained models."""

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

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
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
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

