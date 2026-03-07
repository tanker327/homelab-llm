#!/usr/bin/env python3
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="none")
messages = []

print("Chat with Qwen3.5-35B-A3B (type 'quit' to exit, 'clear' to reset)\n")

while True:
    try:
        user_input = input("\033[1;32mYou:\033[0m ")
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if user_input.strip().lower() == "quit":
        break
    if user_input.strip().lower() == "clear":
        messages.clear()
        print("-- conversation cleared --")
        continue
    if not user_input.strip():
        continue

    messages.append({"role": "user", "content": user_input})

    print("\033[1;34mAssistant:\033[0m ", end="", flush=True)
    try:
        stream = client.chat.completions.create(
            model="Qwen3.5-35B-A3B-Q4_K_M.gguf",
            messages=messages,
            stream=True,
        )
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                reply += delta
        print()
        messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        print(f"\nError: {e}")
