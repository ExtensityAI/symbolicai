# Neuro-Symbolic Engine

The **neuro-symbolic** engine is our generic wrapper around large language models (LLMs) that support prompts, function/tool calls, vision tokens, token‐counting/truncation, etc.
Depending on which backend you configure (OpenAI/GPT, Claude, Deepseek, llama.cpp, HuggingFace, …), a few things must be handled differently:

* GPT-family (OpenAI) and most backends accept the usual `max_tokens`, `temperature`, etc., out of the box.
* Claude (Anthropic) and Deepseek can return an internal “thinking trace” when you enable it.
* Local engines (llamacpp, HuggingFace) do *not* yet support token counting, JSON format enforcement, or vision inputs in the same way.
* Token‐truncation and streaming are handled automatically but may vary in behavior by engine.

---

## Basic Query

```python
from symai import Symbol, Expression

# A one-off question
res = Symbol("Hello, world!").query("Translate to German.")
print(res.value)
# → "Hallo, Welt!"
```

Under the hood this uses the `neurosymbolic` engine.

---

## Raw LLM Response

If you need the raw LLM objects (e.g. `openai.ChatCompletion` or `anthropic.types.Message`), use `raw_output=True`:

```python
from symai import Expression

raw = Expression.prompt("What is the capital of France?", raw_output=True)
# raw.value is the LLM response object
```

---

## Function/Tool Calls

Models that support function calls (OpenAI GPT-4, Claude, …) can dispatch to your `symai.components.Function` definitions:

```python
from symai.components import Function

tools = [{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current temperature for a location.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string"}
      },
      "required": ["location"]
    }
  }
}]

fn = Function(
  "Choose and call the appropriate function",
  tools=tools
)

# GPT-style tool call
resp = fn("What's the temperature in Bogotá, Colombia?", raw_output=True)
# resp.choices[0].finish_reason == "tool_calls"
# resp.choices[0].message.tool_calls[0].function.name == "get_weather"
```

For Claude the API shapes differ slightly:

```python
from anthropic.types import ToolUseBlock

tools = [
  {
    "name": "get_stock_price",
    "description": "Get stock price for a ticker.",
    "input_schema": { … }
  }
]
fn = Function("Pick a function", tools=tools)

# Enable thinking trace (see next section) if needed
resp = fn("What's the S&P 500 today?", raw_output=True, thinking=thinking, max_tokens=16000)
blocks = [b for b in resp.content if isinstance(b, ToolUseBlock)]
assert blocks[0].name == "get_stock_price"
```

---

## Thinking Trace (Claude & Deepseek)

Some engines (Anthropic’s Claude, Deepseek) can return an internal **thinking trace** that shows how they arrived at an answer. To get it, you must:

1. Pass `return_metadata=True`.
2. Pass a `thinking=` configuration if required.
3. Inspect `metadata["thinking"]` after the call.

### Claude (Anthropic)

```python
from symai import Symbol

thinking = {"type": "enabled", "budget_tokens": 4092}

res, metadata = Symbol("Topic: Disneyland") \
    .query(
      "Write a dystopic take on the topic.",
      return_metadata=True,
      thinking=thinking,
      max_tokens=16_000
    )
print(res.value)
print(metadata["thinking"])
```

### Deepseek

```python
from symai import Symbol

# Deepseek returns thinking by default if you enable metadata
res, metadata = Symbol("Topic: Disneyland") \
    .query(
      "Write a dystopic take on the topic.",
      return_metadata=True
    )
print(res.value)
print(metadata["thinking"])
```

---

## JSON‐Only Responses

To force the model to return valid JSON and have `symai` validate it:

```python
import json
from symai import Expression

admin_role = "system"  # or "developer" for non-GPT backends

resp = Expression.prompt(
  message=[
    {"role": admin_role, "content": "You output valid JSON."},
    {"role": "user", "content": (
       "Who won the world series in 2020?"
       " Return as { 'team':'str','year':int,'coach':'str' }"
    )}
  ],
  response_format={"type": "json_object"},
  suppress_verbose_output=True
)
data = json.loads(resp.value)
# data == {"team":"Los Angeles Dodgers", "year":2020, "coach":"Dave Roberts"}
```

---

## Token Counting & Truncation

The default pipeline will automatically estimate token usage and truncate conversation as needed.
On GPT-family backends, raw API usage in `response.usage` matches what `symai` computes.
For Claude / llama.cpp / HuggingFace, skip token‐comparison tests as they are not uniformly supported yet.

---

## Preview Mode

When building new `Function` objects, preview mode lets you inspect the *prepared* prompt **before** calling the engine:

```python
from symai.components import Function

preview_fn = Function(
  "Return a JSON markdown string.",
  static_context="Static…",
  dynamic_context="Dynamic…"
)
# Instead of executing, retrieve prop.processed_input
preview = preview_fn("Hello, World!", preview=True)
assert preview.prop.processed_input == "Hello, World!"
```

---

## Self-Prompting

If you want the model to “self-prompt” (i.e. use the original symbol text as context):

```python
from symai import Symbol

sym = Symbol("np.log2(2)", self_prompt=True)
res = sym.query("Is this equal to 1?", self_prompt=True)
assert "yes" in res.value.lower()
```

---

## Vision Inputs

Vision tokens (e.g. `<<vision:path/to/cat.jpg:>>`) can be passed in prompts on supported backends:

```python
from pathlib import Path
from symai import Symbol

file = Path("assets/images/cat.jpg")
res = Symbol(f"<<vision:{file}:>>").query("What is in the image?")
assert "cat" in res.value.lower()
```

Skip this on llama/HuggingFace if not implemented.

---

## Summary

The neuro-symbolic engine is your central entry point for all LLM-based operations in `symai`.
By tweaking flags such as `return_metadata`, `raw_output`, `preview`, and `thinking`, you can:

- extract low-level API objects,
- retrieve internal thought traces,
- preview prompts before hitting the API,
- enforce JSON formats,
- handle function/tool calls,
- feed in vision data,
- and let `symai` manage token‐truncation automatically.

Refer to the individual engine docs (OpenAI, Claude, Deepseek, llama.cpp, HuggingFace) for any model-specific quirks.
