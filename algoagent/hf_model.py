from __future__ import annotations

import re

from algoagent.model_client import ModelResponse
from algoagent.schema import ProblemSpec


SOLVER_SYSTEM_PROMPT = (
    "You are AlgoAgent, a Python competitive programming assistant. "
    "Answer with Chinese Solution Explanation, time/space complexity, and Python 3 code."
)

SFT_SYSTEM_PROMPT = (
    "You are AlgoAgent, a Python algorithm problem assistant. "
    "Read the Task Type in the user message before answering. "
    "For SOLVE_PROBLEM, output Chinese Solution Explanation, time/space complexity, and Python 3 code. "
    "For ANSWER_FOLLOW_UP, answer the user's follow-up question in Chinese using only the provided verified solution context."
)

FOLLOWUP_SYSTEM_PROMPT = (
    "You are AlgoAgent. Answer follow-up questions only from the verified solution context. "
    "Do not modify code unless the solve-and-verify workflow is restarted."
)


class HuggingFaceModel:
    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: str = "",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        load_in_4bit: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install training dependencies before using HuggingFaceModel.") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        kwargs = {"device_map": "auto", "trust_remote_code": True, "torch_dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model.eval()

    def generate_solution(self, problem: ProblemSpec, feedback: str | None, attempt: int) -> ModelResponse:
        prompt = (
            f"{problem.prompt()}\n"
            f"Attempt: {attempt}\n"
            "Use this exact structure:\n"
            "Solution Explanation:\n...\n\n"
            "Time Complexity: O(...)\n"
            "Space Complexity: O(...)\n"
            "```python\n...\n```\n"
        )
        if feedback:
            prompt += f"\nPrevious feedback:\n{feedback[:4000]}\n"
        text = self._generate(SOLVER_SYSTEM_PROMPT, prompt)
        return ModelResponse(
            raw_text=text,
            explanation=_extract_explanation(text),
            time_complexity=_extract_complexity(text, "time"),
            space_complexity=_extract_complexity(text, "space"),
            code=_extract_python_code(text),
        )

    def answer_followup(
        self,
        problem: ProblemSpec,
        verified_code: str,
        explanation: str,
        complexity: str,
        history: list[tuple[str, str]],
        question: str,
    ) -> str:
        history_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-4:])
        prompt = (
            f"{problem.prompt()}\n\n"
            f"Verified code with line numbers:\n{verified_code}\n\n"
            f"Existing explanation:\n{explanation}\n\n"
            f"Complexity:\n{complexity}\n\n"
            f"History:\n{history_text}\n\n"
            f"User question:\n{question}\n"
        )
        return self._generate(FOLLOWUP_SYSTEM_PROMPT, prompt).strip()

    def _generate(self, system: str, prompt: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature
        with self.torch.no_grad():
            output_ids = self.model.generate(**kwargs)
        generated = output_ids[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def _extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python|py)\s*(.*?)```", text, flags=re.I | re.S)
    return match.group(1).strip() if match else ""


def _extract_explanation(text: str) -> str:
    match = re.search(r"solution\s+explanation\s*:\s*(.*?)(?:time\s+complexity\s*:|$)", text, re.I | re.S)
    return match.group(1).strip() if match else ""


def _extract_complexity(text: str, kind: str) -> str:
    match = re.search(rf"{kind}\s+complexity\s*:\s*(O\s*\([^)]+\)|unknown)", text, re.I)
    return match.group(1) if match else "unknown"
