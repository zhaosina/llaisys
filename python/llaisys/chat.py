from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Sequence

from transformers import AutoTokenizer

import llaisys


class ChatEngine:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = str(model_path)
        self.device_name = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = llaisys.models.Qwen2(self.model_path, _llaisys_device(device))
        self.model_name = Path(self.model_path).name

    def prompt_tokens(self, messages: Sequence[dict[str, str]]) -> list[int]:
        prompt = self.tokenizer.apply_chat_template(
            conversation=list(messages),
            add_generation_prompt=True,
            tokenize=False,
        )
        return self.tokenizer.encode(prompt)

    def generate_tokens(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int = 256,
        top_k: int = 50,
        top_p: float = 0.8,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> tuple[list[int], list[int]]:
        prompt_tokens = self.prompt_tokens(messages)
        output_tokens = self.model.generate(
            prompt_tokens,
            max_new_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
        )
        return prompt_tokens, output_tokens[len(prompt_tokens):]

    def stream_text(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int = 256,
        top_k: int = 50,
        top_p: float = 0.8,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> Iterable[str]:
        prompt_tokens = self.prompt_tokens(messages)
        generated: list[int] = []
        previous_text = ""
        for token in self.model.generate_stream(
            prompt_tokens,
            max_new_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
        ):
            generated.append(token)
            current_text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if current_text.startswith(previous_text):
                delta = current_text[len(previous_text):]
            else:
                delta = current_text
            previous_text = current_text
            if delta:
                yield delta


def completion_response(model_name: str, content: str, prompt_tokens: int, completion_tokens: int) -> dict:
    created = int(time.time())
    return {
        "id": f"chatcmpl-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _llaisys_device(device: str) -> llaisys.DeviceType:
    if device == "cpu":
        return llaisys.DeviceType.CPU
    if device == "nvidia":
        return llaisys.DeviceType.NVIDIA
    raise ValueError(f"Unsupported device: {device}")
