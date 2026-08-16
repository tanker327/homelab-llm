#!/usr/bin/env python3
"""Interactive CLI chat against the production server on :5000.

Commands: quit, clear, effort <low|medium|xhigh|off>.

Per the 2026-08-16 reasoning-effort sweep (docs/BENCHMARKS.md): low/medium
match xhigh's pass rate on coding tasks at ~1/10th the tokens and latency,
so the client defaults to medium instead of the chat template's xhigh.
Reasoning is echoed back into history as `reasoning_content` so the
template's preserve_thinking (default on) actually works across turns —
the OpenAI SDK drops it otherwise.
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="none")
messages = []
effort = "medium"

# Official Qwen3.8 sampling — thinking and non-thinking modes differ.
THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95}
NOTHINK_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5}

print("Chat with Qwen3.8-27B-FP8 (quit, clear, effort <low|medium|xhigh|off>)\n")

while True:
    try:
        user_input = input("\033[1;32mYou:\033[0m ")
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    cmd = user_input.strip().lower()
    if cmd == "quit":
        break
    if cmd == "clear":
        messages.clear()
        print("-- conversation cleared --")
        continue
    if cmd.startswith("effort"):
        parts = cmd.split()
        if len(parts) == 2 and parts[1] in ("low", "medium", "xhigh", "off"):
            effort = parts[1]
            print(f"-- reasoning effort: {effort} --")
        else:
            print(f"-- effort is '{effort}'; usage: effort <low|medium|xhigh|off> --")
        continue
    if not user_input.strip():
        continue

    messages.append({"role": "user", "content": user_input})

    if effort == "off":
        sampling = dict(NOTHINK_SAMPLING)
        sampling["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    else:
        sampling = dict(THINKING_SAMPLING)
        sampling["extra_body"] = {
            "top_k": 20,
            "chat_template_kwargs": {"reasoning_effort": effort},
        }

    print("\033[1;34mAssistant:\033[0m ", end="", flush=True)
    try:
        stream = client.chat.completions.create(
            model="local",
            messages=messages,
            stream=True,
            **sampling,
        )
        reply = ""
        reasoning = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            # vLLM streams thinking as a `reasoning` extra field
            thinking = getattr(delta, "reasoning", None)
            if thinking:
                reasoning += thinking
            if delta.content:
                print(delta.content, end="", flush=True)
                reply += delta.content
        print()
        assistant_msg = {"role": "assistant", "content": reply}
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
        messages.append(assistant_msg)
    except Exception as e:
        print(f"\nError: {e}")
