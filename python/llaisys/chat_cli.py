from __future__ import annotations

import argparse
import json

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8000", type=str)
    parser.add_argument("--model", default="llaisys-chat", type=str)
    parser.add_argument("--max_tokens", default=256, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    messages: list[dict[str, str]] = []
    print("Commands: /reset, /exit")
    while True:
        user_input = input("you> ").strip()
        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/reset":
            messages.clear()
            print("assistant> conversation cleared")
            continue

        messages.append({"role": "user", "content": user_input})
        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "stream": args.stream,
        }

        if args.stream:
            print("assistant> ", end="", flush=True)
            full_text = ""
            with requests.post(f"{args.server}/v1/chat/completions", json=payload, stream=True, timeout=None) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    delta = event["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_text += content
                        print(content, end="", flush=True)
            print()
        else:
            response = requests.post(f"{args.server}/v1/chat/completions", json=payload, timeout=None)
            response.raise_for_status()
            full_text = response.json()["choices"][0]["message"]["content"]
            print(f"assistant> {full_text}")

        messages.append({"role": "assistant", "content": full_text})


if __name__ == "__main__":
    main()
