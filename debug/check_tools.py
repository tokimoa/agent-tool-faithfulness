"""Inspect native tool-calling chat templates for the models we use.

Loads only the tokenizer (fast) and renders a tools= prompt so we can see each
family's expected tool-call output format before implementing the parser.

    uv run --extra mlx python debug/check_tools.py
"""

from __future__ import annotations

from transformers import AutoTokenizer

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "List available result ids and snippets.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": "Read the full text of a result by its id (e.g. r1).",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
]

MODELS = [
    "mlx-community/Qwen3.5-9B-MLX-4bit",
    "mlx-community/Phi-4-mini-instruct-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
]


def main() -> None:
    msgs = [{"role": "user", "content": "Find the calibration code for X."}]
    for mid in MODELS:
        print("\n" + "=" * 70)
        print(mid)
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            out = tok.apply_chat_template(
                msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True
            )
            print(f"TOOLS SUPPORTED (rendered {len(out)} chars). Tail:")
            print(out[-700:])
        except Exception as e:  # noqa: BLE001
            print("tools NOT supported / error:", repr(e))


if __name__ == "__main__":
    main()
