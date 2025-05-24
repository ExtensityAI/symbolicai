# Neuro-Symbolic Engine

The **neuro-symbolic** engine is our generic wrapper around large language models (LLMs) that support prompts, function/tool calls, vision tokens, token‐counting/truncation, etc.
Depending on which backend you configure (OpenAI/GPT, Claude, Gemini, Deepseek, llama.cpp, HuggingFace, …), a few things must be handled differently:

* GPT-family (OpenAI) and most backends accept the usual `max_tokens`, `temperature`, etc., out of the box.
* Claude (Anthropic), Gemini (Google), and Deepseek can return an internal “thinking trace” when you enable it.
* Local engines (llamacpp, HuggingFace) do *not* yet support token counting, JSON format enforcement, or vision inputs in the same way.
* Token‐truncation and streaming are handled automatically but may vary in behavior by engine.

**Note**: the most accurate documentation is the _code_, so be sure to check out the tests. Look for the `mandatory` mark since those are the features that were tested and are guaranteed to work.

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

If you need the raw LLM objects (e.g. `openai.ChatCompletion`, `anthropic.types.Message`/`anthropic.Stream`, or `google.genai.types.GenerateContentResponse`), use `raw_output=True`:

```python
from symai import Expression

raw = Expression.prompt("What is the capital of France?", raw_output=True)
# raw.value is the LLM response object
```

---

## Function/Tool Calls

Models that support function calls (OpenAI GPT-4, Claude, Gemini, …) can dispatch to your `symai.components.Function` definitions:

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

## Thinking Trace (Claude, Gemini & Deepseek)

Some engines (Anthropic's Claude, Google's Gemini, Deepseek) can return an internal **thinking trace** that shows how they arrived at an answer. To get it, you must:

1. Pass `return_metadata=True`.
2. Pass a `thinking=` configuration if required.
3. Inspect `metadata["thinking"]` after the call.

### Claude (Anthropic)

```python
from symai import Symbol

thinking = {"budget_tokens": 4092}

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

### Gemini (Google)

```python
from symai import Symbol

thinking = {"thinking_budget": 1024}

res, metadata = Symbol("Topic: Disneyland") \
    .query(
      "Write a dystopic take on the topic.",
      return_metadata=True,
      thinking=thinking
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
For Gemini, an API call is made to retrieve token counts.
For Claude / llama.cpp / HuggingFace, skip token‐comparison tests as they are not uniformly supported yet.

If a tokenizer is available for the current engine, you can easily count tokens in a string via `Symbol`:
```python
string = "Hello, World!"
print(Symbol(string).tokens)
```

### Tracking Usage and Estimating Costs with `MetadataTracker`

For more detailed tracking of API calls, token usage, and estimating costs, you can use the `MetadataTracker` in conjunction with `RuntimeInfo`. This is particularly useful for monitoring multiple calls within a specific code block.
> Note: we only track OpenAI models for now (chat and search).

`MetadataTracker` collects metadata from engine calls made within its context. `RuntimeInfo` then processes this raw metadata to provide a summary of token counts, number of API calls, elapsed time, and an estimated cost if pricing information is provided.

Here's an example of how to use them:

```python
import time
from symai import Symbol, Interface
from symai.components import MetadataTracker
from symai.utils import RuntimeInfo
from symai.backend.settings import SYMAI_CONFIG

# This is a simplified cost estimation function.
# It's called for each engine's usage data.
def estimate_cost_for_engine(info: RuntimeInfo, pricing: dict) -> float:
    # Cost calculation including cached tokens and per-call costs.
    # Assumes 'pricing' dict contains 'input', 'cached_input', 'output', and optionally 'calls' keys.
    input_cost = (info.prompt_tokens - info.cached_tokens) * pricing.get("input", 0)
    cached_input_cost = info.cached_tokens * pricing.get("cached_input", 0)
    output_cost = info.completion_tokens * pricing.get("output", 0)
    call_cost = info.total_calls * pricing.get("calls", 0)  # Cost for the number of API calls
    return input_cost + cached_input_cost + output_cost + call_cost

# This check is illustrative; adapt as needed for your environment.
NEUROSYMBOLIC_ENGINE_IS_OPENAI = 'gpt' in SYMAI_CONFIG.get('NEUROSYMBOLIC_ENGINE_MODEL', '').lower()
SEARCH_ENGINE_IS_OPENAI = 'gpt' in SYMAI_CONFIG.get('SEARCH_ENGINE_MODEL', '').lower()

if NEUROSYMBOLIC_ENGINE_IS_OPENAI and SEARCH_ENGINE_IS_OPENAI:
    sym = Symbol("This is a context sentence.")
    # We'll assume 'openai_search' interface maps to an engine like 'GPTXSearchEngine' or similar.
    # This engine name should then be a key in your dummy_pricing dictionary.
    search = Interface('openai_search')
    start_time = time.perf_counter()
    with MetadataTracker() as tracker:
        res = sym.query("What is the capital of France?")
        search_res = search("What are the latest developments in AI?")
    end_time = time.perf_counter()

    # Dummy pricing for cost estimation.
    # Keys are tuples of (engine_name, model_id) and should match the keys
    # that appear in the tracker.usage dictionary. These depend on your SYMAI_CONFIG
    # and the models used by the engines.
    dummy_pricing = {
        ("GPTXChatEngine", "gpt-model-A"): { # Example: engine used by sym.query() and its model
            "input": 0.000002,
            "cached_input": 0.000001,
            "output": 0.000002
        },
        ("GPTXSearchEngine", "gpt-model-B"): { # Example: engine used by Interface('openai_search') and its model
            "input": 0.000002,
            "cached_input": 0.000001,
            "output": 0.000002,
            "calls": 0.0001 # Cost per API call
        }
        # Add other (engine_name, model_id) tuples and their pricing if used.
    }

    # Process collected data:
    # RuntimeInfo.from_tracker returns a dictionary where keys are (engine_name, model_id) tuples
    # and values are RuntimeInfo objects for each engine-model combination.
    # We pass 0 for total_elapsed_time initially, as it's set for the aggregated sum later.
    usage_per_engine = RuntimeInfo.from_tracker(tracker, 0)

    # Initialize an empty RuntimeInfo object to aggregate totals
    aggregated_usage = RuntimeInfo(
        total_elapsed_time=0,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        cached_tokens=0,
        total_calls=0,
        total_tokens=0,
        cost_estimate=0
    )
    for (engine_name, model_id), engine_data in usage_per_engine.items():
        pricing_key = (engine_name, model_id)
        if pricing_key in dummy_pricing:
            # Estimate cost for this specific engine and model
            engine_data_with_cost = RuntimeInfo.estimate_cost(engine_data, estimate_cost_for_engine, pricing=dummy_pricing[pricing_key])
            aggregated_usage += engine_data_with_cost # Aggregate data
    # Set the total elapsed time for the aggregated object
    aggregated_usage.total_elapsed_time = end_time - start_time
```

This approach provides a robust way to monitor and control costs associated with LLM API usage, especially when making multiple calls. Remember to update the `pricing` dictionary with the current rates for the models you are using. The `estimate_cost` function can also be customized to reflect complex pricing schemes (e.g., different rates for different models, image tokens, etc.).

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

Refer to the individual engine docs (OpenAI, Claude, Gemini, Deepseek, llama.cpp, HuggingFace) for any model-specific quirks.
