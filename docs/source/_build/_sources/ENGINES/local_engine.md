# Local Engines
## Local Neuro-Symbolic Engine

You can use a locally hosted instance for the Neuro-Symbolic Engine. We build on top of:
- [llama.cpp](https://github.com/ggerganov/llama.cpp/tree/master) either through:
  - Direct C++ server from llama.cpp
  - [llama-cpp-python](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file)
- [huggingface/transformers](https://huggingface.co/docs/transformers/en/index) through a custom FastAPI server.

### llama.cpp backend
For instance, let's suppose you want to set as a Neuro-Symbolic Engine the latest Llama 3 model. First, download the model with the HuggingFace CLI:
```bash
huggingface-cli download TheBloke/LLaMA-Pro-8B-Instruct-GGUF llama-pro-8b-instruct.Q4_K_M.gguf --local-dir .
```

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
symserver --env cpp --cpp-server-path /path/to/llama.cpp/server -m ./llama-pro-8b-instruct.Q4_K_M.gguf --port 8000 --host localhost
```

To see all available options, run:
```bash
symserver --env python --help  # for Python bindings
symserver --env cpp --cpp-server-path /path/to/llama.cpp/server --help  # for C++ server
```

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
symserver --env cpp --cpp-server-path /path/to/llama.cpp/server  -ngl 0 -m nomic-embed-text-v1.5.Q8_0.gguf --embedding -c 8192 -b 8192 --rope-scaling yarn --rope-freq-scale .75 --port 8000 --host localhost
```

The server supports batch processing for embeddings. Here's how to use it with `symai`:

```python
from symai import Symbol

# Single text embedding
some_text = "Hello, world!"
embedding = Symbol(some_text).embed()  # returns a list (1 x dim)

# Batch processing
some_batch_of_texts = ["Hello, world!"] * 32
embeddings = Symbol(some_batch_of_texts).embed()  # returns a list (32 x dim)
```
