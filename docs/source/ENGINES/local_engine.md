# Local Engines
## Local Neuro-Symbolic Engine

You can use a locally hosted instance for the Neuro-Symbolic Engine. We build on top of:
- [llama.cpp](https://github.com/ggerganov/llama.cpp/tree/master) either through:
    > ❗️**NOTE**❗️ Latest `llama.cpp` commit on `master` branch on November 5th, 2025 that we tested `symai` with is `a5c07dcd7b49`. We used the build [setup](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
  - Direct C++ server from llama.cpp
  - [llama-cpp-python](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file)
- [huggingface/transformers](https://huggingface.co/docs/transformers/en/index) through a custom FastAPI server.

### llama.cpp backend
For instance, let's suppose you want to set up the Neuro-Symbolic Engine with the `gpt-oss-120b` model. Download the GGUF shards you need (e.g. the `Q4_1` variant).

With `symai`, first set the `NEUROSYMBOLIC_ENGINE_MODEL` to `llamacpp`:

```json
{
  "NEUROSYMBOLIC_ENGINE_API_KEY": "",
  "NEUROSYMBOLIC_ENGINE_MODEL": "llamacpp",
  ...
}
```

You can then run the server in two ways:

1. Using Python bindings:
```bash
symserver --env python --model ./llama-pro-8b-instruct.Q4_K_M.gguf --n_gpu_layers -1 --chat_format llama-3 --port 8000 --host localhost
```

2. Using C++ server directly:
```bash
symserver --env cpp --cpp-server-path /path/to/llama.cpp/llama-server -ngl -1 -m gpt-oss-120b/Q4_1/gpt-oss-120b-Q4_1-00001-of-00002.gguf -fa 'on' -b 8092 -ub 1024 --port 8000 --host localhost -c 0 -n 4096 -t 14 --jinja
```

To see all available options, run:
```bash
symserver --env python --help  # for Python bindings
symserver --env cpp --cpp-server-path /path/to/llama.cpp/llama-server --help  # for C++ server
```

The Neuro-Symbolic Engine now supports tool execution and structured JSON responses out of the box. For concrete examples, review the tests in `tests/engines/neurosymbolic/test_nesy_engine.py::test_tool_usage` and `tests/contract/test_contract.py`.

### vLLM backend (experimental)

> ⚠️ **Experimental.** This integration is provided on a best-effort basis. The
> wire layer (symai engine ↔ vLLM's OpenAI-compatible API) is verified, but
> vLLM's CPU backend itself is fragile on macOS Apple Silicon (no pre-built
> wheels, `Qwen3_5MoeForConditionalGeneration` family does not load on CPU,
> V1 `EngineCore` has a 60 s shm-broadcast timeout that trips on
> reasoning-heavy generations). Expect breakage on non-CUDA hosts and treat
> this as a preview — not something to base a production workflow on. For
> actual local use on Apple Silicon, prefer the llama.cpp backend; reserve
> this path for GPU hosts or for wiring / protocol testing.

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput OpenAI-compatible inference server. `symai` drives it the same way as the llama.cpp backend: `symserver` launches the vLLM server as a subprocess, and the neuro-symbolic engine talks to it over HTTP on `/v1/chat/completions`.

Set the `NEUROSYMBOLIC_ENGINE_MODEL` to `vllm`:

```json
{
  "NEUROSYMBOLIC_ENGINE_API_KEY": "",
  "NEUROSYMBOLIC_ENGINE_MODEL": "vllm",
  ...
}
```

#### Install: clone + source build (the supported path on every platform)

vLLM does not publish pre-built wheels for macOS, and its source build on macOS requires specific flags that uv's default resolver does not supply. The reliable, cross-platform path — and the one we actively tested against — is the same layout as the llama.cpp C++ backend: clone vLLM into its own directory with its own venv, build it there, and point `symserver` at that interpreter via `--vllm-python-path` (analogous to `--cpp-server-path` for llama.cpp).

> ❗️**Tested against**❗️ vLLM main branch at commit
> [`595562651a5a4539ffa910d8570c08fb5169bdc9`](https://github.com/vllm-project/vllm/commit/595562651a5a4539ffa910d8570c08fb5169bdc9)
> (`0.1.dev50+g595562651`, ~50 commits past the `v0.11.0` tag). Newer commits
> may work but are untested. Older commits (≤ `v0.11.0`) do **not** recognise
> newer Qwen3 MoE architectures (e.g. `Qwen3_5MoeForConditionalGeneration`)
> and will reject them at load time.

Requirements for the source build:
- macOS Sonoma or later, **or** Linux
- Python 3.11 or 3.12
- `uv` ≥ 0.9 (installs itself via `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **macOS:** XCode Command Line Tools with Apple Clang ≥ 15.0
- **Linux:** `gcc/g++ ≥ 12.3.0`

One-time setup (separate venv outside `symai`):

```bash
git clone https://github.com/vllm-project/vllm.git ~/Devspace/projects/vllm
cd ~/Devspace/projects/vllm
git checkout 595562651a5a4539ffa910d8570c08fb5169bdc9      # pin to the tested commit
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
VIRTUAL_ENV=$PWD/.venv uv pip install -e .                 --index-strategy unsafe-best-match
```

The `--index-strategy unsafe-best-match` flag is required by vLLM's official build docs — it lets `uv` pull torch from the PyTorch index and everything else from PyPI in a single resolution. Without it, you hit `typing-extensions` conflicts and `uv pip install` returns happily while the native extension stays unbuilt. macOS auto-sets `VLLM_TARGET_DEVICE=cpu` inside the build.

Build troubleshooting:
- `'map' file not found` / other missing C++ headers → reinstall XCode Command Line Tools.
- C++11/C++17 errors → edit `cmake/cpu_extension.cmake` and add `set(CMAKE_CXX_STANDARD 17)` before `set(CMAKE_CXX_STANDARD_REQUIRED ON)`.
- `CUDA_HOME not set` while installing into the symai venv directly → you're in the wrong venv; run the steps above in the **vllm project's** venv.

Once installed, confirm the native extension loaded correctly:

```bash
~/Devspace/projects/vllm/.venv/bin/python -c \
  "import vllm, torch; torch.ops._C.silu_and_mul; print('vllm:', vllm.__version__, 'ops OK')"
```

Then, from the `symai` project, run `symserver` pointing at that interpreter:

```bash
symserver --vllm-python-path ~/Devspace/projects/vllm/.venv/bin/python \
          --model Qwen/Qwen3-4B-Instruct-2507 \
          --host localhost --port 8000 \
          --dtype float16 \
          --max-model-len 4096 \
          --max-num-seqs 1 \
          --enforce-eager \
          --no-enable-prefix-caching
```

`vllm serve` auto-downloads the model from HuggingFace Hub on first use and honors `HF_HOME` / `HUGGINGFACE_HUB_CACHE`.

#### Tool calling

No tool-call parser is baked into `symserver`. Pass the parser that matches the model at launch time:

```bash
symserver --vllm-python-path ~/Devspace/projects/vllm/.venv/bin/python \
          --model Qwen/Qwen3-4B-Instruct-2507 \
          --host localhost --port 8000 \
          --dtype float16 --max-model-len 4096 \
          --enable-auto-tool-choice \
          --tool-call-parser hermes
```

#### Reasoning / thinking traces

If you want `metadata['thinking']` populated when calling the engine with `return_metadata=True`, launch with `--reasoning-parser`:

```bash
symserver --vllm-python-path ~/Devspace/projects/vllm/.venv/bin/python \
          --model Qwen/Qwen3-4B-Thinking-2507 \
          --host localhost --port 8000 \
          --dtype float16 --max-model-len 4096 \
          --reasoning-parser qwen3
```

The engine reads `choices[*].message.reasoning_content` and copies it into `metadata['thinking']`.

#### Seeing every vLLM flag

Any flag not consumed by `symserver` itself (`--help`, `--entrypoint`, `--vllm-python-path`) is forwarded verbatim, so the full server surface — `--tensor-parallel-size`, `--gpu-memory-utilization`, `--served-model-name`, `--chat-template`, `--enforce-eager`, `--trust-remote-code`, etc. — is available.

```bash
symserver --vllm-python-path ~/Devspace/projects/vllm/.venv/bin/python --help
```

#### Using it from Python

Once the server is up, using the engine is transparent:

```python
from symai import Symbol

sym = Symbol('Kitties are cute!').compose()
print(sym)
```

#### Caveats — honest ones

- **No macOS PyPI wheels.** `pip install vllm` on macOS will attempt a source build that defaults to the CUDA extension; do **not** do this inside the symai venv. The clone-and-build workflow above is the only clean path.
- **macOS CPU supports only FP32 and FP16.** No BF16. Pass `--dtype float16` or `--dtype float32`. Memory-wise, a 36B BF16 model like `Qwen/Qwen3.6-35B-A3B` is ~72 GB at FP16 and fits on a 128 GB Apple Silicon, but…
- **Qwen3.5-MoE architecture family is currently broken on vLLM CPU.** Models whose `architectures` field is `Qwen3_5MoeForConditionalGeneration` (e.g. `Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.5-35B-A3B`) load but fail in `process_weights_after_loading` — vLLM's CPU GEMM dispatcher expects 2-D linear weights, and MoE expert tensors are 3-D. Use a dense model (`Qwen3ForCausalLM`, `LlamaForCausalLM`, …) until upstream ships a fix.
- **vLLM V1 CPU backend is fragile for long generations.** The shared-memory broadcast between `EngineCore` and `WorkerProc` uses a 60 s per-step timeout. On CPU, a single decode step can exceed that for reasoning-heavy models emitting long `<think>` traces, and the engine will declare the worker dead and shut down. For long outputs bump the timeout:
  ```bash
  VLLM_SHM_RING_BUFFER_WAIT_SECS=600 symserver --vllm-python-path …
  ```
- **Performance on Apple Silicon is poor.** vLLM's CPU backend primarily targets x86 AVX-512, falls back to generic PyTorch ops on ARM, and has no Metal backend. Per-token latency on macOS CPU is typically 3-10× slower than `llama.cpp` on the same hardware. vLLM's value is GPU serving with continuous batching; on a single-user macOS workstation, prefer the llama.cpp backend for actual use and reserve vLLM for CUDA hosts.

### HuggingFace backend
Let's suppose we want to use `dolphin-2.9.3-mistral-7B-32k` from HuggingFace. First, download the model with the HuggingFace CLI:
```bash
huggingface-cli download cognitivecomputations/dolphin-2.9.3-mistral-7B-32k --local-dir ./dolphin-2.9.3-mistral-7B-32k
```

For the HuggingFace server, you have to set the `NEUROSYMBOLIC_ENGINE_MODEL` to `huggingface`:
```json
{
  "NEUROSYMBOLIC_ENGINE_API_KEY": "",
  "NEUROSYMBOLIC_ENGINE_MODEL": "huggingface",
  ...
}
```

Then, run `symserver` with the following options:
```bash
symserver --model ./dolphin-2.9.3-mistral-7B-32k --attn_implementation flash_attention_2
```

To see all the available options we support for HuggingFace, run:
```bash
symserver --help
```

Now you are set to use the local engine.

```python
# do some symbolic computation with the local engine
sym = Symbol('Kitties are cute!').compose()
print(sym)

# :Output:
# Kittens are known for their adorable nature and fluffy appearance, making them a favorite addition to many homes across the world. They possess a strong bond with
# their owners, providing companionship and comfort that can ease stress and anxiety. With their playful personalities, they are often seen as a symbol of happiness
# and joy, and their unique characteristics such as purring, kneading, and head butts bring warmth to our hearts. Cats also have a natural instinct to groom, which
# helps them maintain their clean and soft fur. Not only do they bring comfort and love to their owners, but they also have some practical benefits, such as reducing
# allergens, deterring pests, and even reducing stress in their surroundings. Overall, it is no surprise that pets have a long history of providing both emotional
# and physical comfort and happiness to their owners, making them a much-loved member of families around the world.
```

## Local Embedding Engine

You can also use local embedding models through the `llama.cpp` backend. First, set the `EMBEDDING_ENGINE_MODEL` to `llamacpp`:

```json
{
  "EMBEDDING_ENGINE_API_KEY": "",
  "EMBEDDING_ENGINE_MODEL": "llamacpp",
  ...
}
```

For instance, to use the Nomic embed text model, first download it:
```bash
huggingface-cli download nomic-ai/nomic-embed-text-v1.5-GGUF nomic-embed-text-v1.5.Q8_0.gguf --local-dir .
```

Then start the server with embedding-specific parameters using either:

Python bindings:
```bash
symserver --env python --model nomic-embed-text-v1.5.Q8_0.gguf --embedding True --n_ctx 2048 --rope_scaling_type 2 --rope_freq_scale 0.75 --n_batch 32 --port 8000 --host localhost
```

C++ server:
```bash
symserver --env cpp --cpp-server-path /path/to/llama.cpp/llama-server -ngl -1 -m nomic-embed-text-v1.5.Q8_0.gguf --embedding -b 8092 -ub 1024 --port 8000 --host localhost -t 14 --mlock --no-mmap
```

The server supports batch processing for embeddings. Here's how to use it with `symai`:

```python
from symai import Symbol

# Single text embedding
some_text = "Hello, world!"
embedding = Symbol(some_text).embed()  # returns a list (1 x dim)

# Batch processing
some_batch_of_texts = ["Hello, world!"] * 32
embeddings = Symbol(some_batch_of_texts).embed()  # returns a list (32 x 1 x dim)
```
