# API Documentation

The server provides an OpenAI-compatible API. Production (since 2026-08-28) is **llama.cpp serving Qwen3.8-Flash-Next-Uncensored** (262K context, vision, thinking mode) on a single endpoint.

**Base URL:** `http://localhost:5000` — from the LAN: `http://192.168.10.106:5000`. (Port 5001 exists only under the manual two-model agent stack.)

**Authentication:** None required (`api_key` can be any string).

**Model name:** always send `"local"`. The llama.cpp production engine accepts any string (it reports the launcher's `--alias local` via `/v1/models`), but the vLLM fallback engines (NVFP4/FP8 27B) **reject** names other than `local` — hardcoding `"local"` works across every engine.

**Engine differences to know:** this doc describes the llama.cpp production engine. When the conf is switched to a vLLM engine: chain-of-thought arrives in `reasoning` instead of `reasoning_content`, `max_tokens` works correctly on chat completions, and the `model` field is validated. Everything else is identical.

---

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completion (main endpoint) |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List available models |
| `/health` | GET | Server health check |

---

## POST /v1/chat/completions

Generate a chat response from a conversation.

### Request Body

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | required | Always `"local"` (see Model name note above) |
| `messages` | array | required | Conversation messages (see [Message Format](#message-format)) |
| `chat_template_kwargs` | object | server pins `medium` | Template overrides, e.g. `{"reasoning_effort": "low"}` to clamp thinking. Only `xhigh`, `medium`, `low` are valid |
| `temperature` | float | 1.0 | Sampling temperature (0.0 = deterministic, higher = more random) |
| `top_p` | float | 0.95 | Nucleus sampling threshold |
| `top_k` | integer | 20 | Top-k sampling (limits to k most likely tokens) |
| `stream` | boolean | false | Stream response as Server-Sent Events |
| `stop` | array | [] | Stop sequences (generation stops when any is produced) |
| `frequency_penalty` | float | 0.0 | Penalize repeated tokens by frequency |
| `presence_penalty` | float | 0.0 | Penalize tokens that already appeared |
| `seed` | integer | null | Random seed for reproducible output |

> **Note:** `max_tokens` is not reliably supported on the chat endpoint due to a known issue with the thinking model's output parser. If generation is truncated mid-thinking, the server returns a 500 error. Omit this field and let the model finish naturally, or use `stop` sequences to control length.

### Message Format

Each message in the `messages` array has:

| Field | Type | Description |
|---|---|---|
| `role` | string | One of: `system`, `user`, `assistant` |
| `content` | string or array | Plain text, or an array of content parts for vision |

For vision (production supports it — the mmproj loads by default), `content` is an array mixing `{"type": "text", "text": ...}` and `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` parts.

### Example Request

```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

### Example Response

```json
{
  "id": "chatcmpl-9CIRokWgbwRyNhk1CFDRphXXPlipmEGu",
  "object": "chat.completion",
  "created": 1772913014,
  "model": "local",
  "system_fingerprint": "b8233-c5a778891",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "2 + 2 = 4.",
        "reasoning_content": "Thinking Process:\n\n1. Analyze the Request...\n"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 228,
    "total_tokens": 256
  },
  "timings": {
    "prompt_n": 28,
    "prompt_ms": 41.598,
    "prompt_per_second": 673.1,
    "predicted_n": 228,
    "predicted_ms": 1365.931,
    "predicted_per_second": 166.9
  }
}
```

### Response Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier for the completion |
| `object` | string | Always `"chat.completion"` |
| `created` | integer | Unix timestamp |
| `model` | string | Model used |
| `choices` | array | Array of completion choices (usually 1) |
| `choices[].index` | integer | Choice index |
| `choices[].finish_reason` | string | `"stop"` (natural end) or `"length"` (hit token limit) |
| `choices[].message.role` | string | Always `"assistant"` |
| `choices[].message.content` | string | The actual response text |
| `choices[].message.reasoning_content` | string | The model's internal thinking/reasoning (chain-of-thought) |
| `usage.prompt_tokens` | integer | Tokens in the prompt |
| `usage.completion_tokens` | integer | Tokens generated (includes thinking + content) |
| `usage.total_tokens` | integer | Sum of prompt + completion tokens |
| `timings.prompt_per_second` | float | Prompt processing speed (tok/s) |
| `timings.predicted_per_second` | float | Generation speed (tok/s) |
| `timings.prompt_ms` | float | Time to process prompt (ms) |
| `timings.predicted_ms` | float | Time to generate response (ms) |

### Thinking / Reasoning

This model has a built-in thinking mode (similar to DeepSeek-R1). The response separates:

- **`content`** — the final answer shown to the user
- **`reasoning_content`** — the model's internal chain-of-thought reasoning

The `reasoning_content` is always populated. Token counts in `usage` include both thinking and content tokens.

---

## POST /v1/chat/completions (Streaming)

Set `"stream": true` to receive Server-Sent Events.

### Example Request

```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

### Stream Format

Each event is a line starting with `data: ` followed by JSON:

```
data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null},"finish_reason":null}],...}

data: {"choices":[{"index":0,"delta":{"reasoning_content":"Thinking"},"finish_reason":null}],...}

data: {"choices":[{"index":0,"delta":{"reasoning_content":" about"},"finish_reason":null}],...}

data: {"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}],...}

data: {"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}],...}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],...}

data: [DONE]
```

**Stream event fields:**

| Field | Description |
|---|---|
| `delta.role` | Set to `"assistant"` in the first chunk only |
| `delta.reasoning_content` | Thinking tokens (streamed before content) |
| `delta.content` | Response content tokens |
| `delta` (empty) | Final chunk, check `finish_reason` |
| `finish_reason` | `null` during generation, `"stop"` when done |
| `timings` | Included in the final chunk (same fields as non-streaming) |

The stream order is: role → reasoning tokens → content tokens → stop.

---

## POST /v1/completions

Raw text completion (no chat formatting).

### Request Body

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | required | Model name |
| `prompt` | string | required | Text prompt to complete |
| `max_tokens` | integer | 256 | Maximum tokens to generate |
| `temperature` | float | 1.0 | Sampling temperature |
| `top_p` | float | 0.95 | Nucleus sampling |
| `top_k` | integer | 20 | Top-k sampling |
| `stream` | boolean | false | Stream response |
| `stop` | array | [] | Stop sequences |
| `echo` | boolean | false | Include prompt in response |
| `seed` | integer | null | Random seed |

### Example Request

```bash
curl http://localhost:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "prompt": "The capital of Japan is",
    "max_tokens": 20,
    "temperature": 0.0
  }'
```

### Example Response

```json
{
  "id": "chatcmpl-iQYfRSVI5TqN9NkoFgYXMsFEvKqKGfgc",
  "object": "text_completion",
  "created": 1772913029,
  "model": "local",
  "choices": [
    {
      "index": 0,
      "text": " Tokyo.\nThe capital of Japan is Tokyo.",
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 20,
    "total_tokens": 25
  },
  "timings": {
    "prompt_per_second": 201.0,
    "predicted_per_second": 164.6,
    "prompt_ms": 24.871,
    "predicted_ms": 121.539
  }
}
```

---

## GET /v1/models

List loaded models.

### Example

```bash
curl http://localhost:5000/v1/models
```

### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "local",
      "object": "model",
      "created": 1787937772,
      "owned_by": "llamacpp",
      "meta": {
        "n_vocab": 248320,
        "n_ctx": 262144,
        "n_ctx_train": 262144,
        "n_embd": 2560,
        "n_params": 176943899520,
        "size": 97462131200,
        "ftype": "IQ4_XS - 4.25 bpw"
      }
    }
  ]
}
```

---

## GET /health

Check server status.

### Response

```json
{"status": "ok"}
```

Possible status values:
- `ok` — ready to serve requests
- `loading model` — model is still loading
- `error` — server error

---

## Client Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/v1",
    api_key="none",  # any string works
)

# Non-streaming
response = client.chat.completions.create(
    model="local",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain gravity in one sentence."},
    ],
    temperature=0.7,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
```

### JavaScript (fetch)

```javascript
const response = await fetch("http://192.168.10.106:5000/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "local",
    messages: [{ role: "user", content: "Hello" }],
  }),
});
const data = await response.json();
console.log(data.choices[0].message.content);
```

### JavaScript (OpenAI SDK)

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://192.168.10.106:5000/v1",
  apiKey: "none",
});

const response = await client.chat.completions.create({
  model: "local",
  messages: [{ role: "user", content: "Hello" }],
});
console.log(response.choices[0].message.content);
```

### curl (Multi-turn Conversation)

```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [
      {"role": "system", "content": "You are a coding assistant."},
      {"role": "user", "content": "Write a hello world in Python"},
      {"role": "assistant", "content": "print(\"Hello, World!\")"},
      {"role": "user", "content": "Now make it a function"}
    ]
  }'
```

---

## Known Limitations

1. **`max_tokens` not supported on `/v1/chat/completions`** — The thinking model's output parser fails when generation is truncated mid-reasoning. Use `stop` sequences or omit the field. The `/v1/completions` endpoint supports `max_tokens` normally.

2. **Token counts include thinking** — `usage.completion_tokens` includes both `reasoning_content` and `content` tokens. Actual visible response is shorter than the token count suggests.

3. **Context window** — The full native 262,144 tokens (prompt + reasoning + output combined) is served, verified by needle retrieval to 261K. Four server slots share a unified KV pool.

4. **Concurrency** — Tuned for 1–4 concurrent users; aggregate throughput plateaus ~130 tok/s at N=4 and queueing grows sharply beyond that. For wide agent fleets, switch `production-engine.conf` to the NVFP4 vLLM engine.

5. **Single model** — Only one model is loaded at a time. The llama.cpp engine ignores the `model` field (but always send `"local"` for vLLM-engine compatibility).

6. **No embeddings endpoint** — llama-server does not serve `/v1/embeddings`.
