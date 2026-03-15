from __future__ import annotations

import argparse
import json
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from .chat import ChatEngine, completion_response


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 50
    stream: bool = False
    seed: int | None = None


def create_app(model_path: str, device: str = "cpu") -> FastAPI:
    state: dict[str, ChatEngine] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["engine"] = ChatEngine(model_path, device=device)
        yield
        state.clear()

    app = FastAPI(title="LLAISYS Chat Server", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": state["engine"].model_name, "device": device}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        engine = state["engine"]
        messages = [message.model_dump() for message in request.messages]

        if request.stream:
            def event_stream():
                created = int(time.time())
                completion_id = f"chatcmpl-{created}"
                prompt_tokens = len(engine.prompt_tokens(messages))
                emitted_tokens = 0
                first = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": engine.model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
                for chunk in engine.stream_text(
                    messages,
                    max_tokens=request.max_tokens,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    temperature=request.temperature,
                    seed=request.seed,
                ):
                    emitted_tokens += 1
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": engine.model_name,
                        "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                final_payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": engine.model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": emitted_tokens,
                        "total_tokens": prompt_tokens + emitted_tokens,
                    },
                }
                yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        prompt_tokens, completion_tokens = engine.generate_tokens(
            messages,
            max_tokens=request.max_tokens,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            seed=request.seed,
        )
        content = engine.tokenizer.decode(completion_tokens, skip_special_tokens=True)
        return JSONResponse(
            completion_response(
                engine.model_name,
                content,
                prompt_tokens=len(prompt_tokens),
                completion_tokens=len(completion_tokens),
            )
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    uvicorn.run(create_app(args.model, device=args.device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
