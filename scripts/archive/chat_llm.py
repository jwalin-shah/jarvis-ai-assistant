#!/usr/bin/env python3
"""Interactive chat with the local LFM model. Type prompts, see responses.

Usage:
    uv run python scripts/chat_llm.py
    uv run python scripts/chat_llm.py --model models/lfm2-1.2b-extract-mlx-4bit
    uv run python scripts/chat_llm.py --max-tokens 200 --temp 0.0
"""
import argparse
import sys

sys.path.insert(0, ".")
from models.loader import MLXModelLoader, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with local LLM")
    parser.add_argument(
        "--model", type=str, default=None, help="Model path (default: LFM2.5-Instruct)"
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--system", type=str, default=None, help="System prompt")
    args = parser.parse_args()

    config = ModelConfig(model_path=args.model) if args.model else ModelConfig()
    loader = MLXModelLoader(config)

    print(f"Loading {config.model_path}...", flush=True)
    loader.load()
    print("Ready! Type your prompt (Ctrl+C to quit)\n", flush=True)

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # Build chat messages
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            messages.append({"role": "user", "content": user_input})

            formatted = loader._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            result = loader.generate_sync(
                prompt=formatted,
                max_tokens=args.max_tokens,
                temperature=args.temp,
                repetition_penalty=1.05,
                timeout_seconds=30.0,
                pre_formatted=True,
            )
            print(f"\nLLM: {result.text}")
            print(f"  [{result.tokens_generated} tokens, {result.generation_time_ms:.0f}ms]\n")

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}")

    loader.unload()


if __name__ == "__main__":
    main()
