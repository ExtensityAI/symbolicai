# Local Neuro-Symbolic Engine

You can use a locally hosted instance for the Neuro-Symbolic Engine. We build on top of:
- [llama.cpp](https://github.com/ggerganov/llama.cpp/tree/master) through [llama-cpp-python](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file). Please follow the `llama-cpp-python` installation instructions. We make the assumption the user has experience running `llama.cpp` prior to using our API for local hosting.
- [huggingface/transformers](https://huggingface.co/docs/transformers/en/index) through a custom FastAPI server.

### llama.cpp backend
For instance, let's suppose you want to set as a Neuro-Symbolic Engine the latest Llama 3 model. First, download the model with the HuggingFace CLI:
```bash
huggingface-cli download TheBloke/LLaMA-Pro-8B-Instruct-GGUF llama-pro-8b-instruct.Q4_K_M.gguf --local-dir .
```

Normally, to start the server through `llama.cpp` you would run something that looks like this:
```bash
python -m llama_cpp.server --model ./llama-pro-8b-instruct.Q4_K_M.gguf --n_gpu_layers -1 --chat_format llama-3 --port 8000 --host localhost
```
With `symai`, simply set the `NEUROSYMBOLIC_ENGINE_MODEL` to `llamacpp`:

```json
{
  "NEUROSYMBOLIC_ENGINE_API_KEY": "",
  "NEUROSYMBOLIC_ENGINE_MODEL": "llamacpp",
  ...
}
```
Then, run `symserver` with options available for `llama.cpp`:
```bash
symserver --model ./llama-pro-8b-instruct.Q4_K_M.gguf --n_gpu_layers -1 --chat_format llama-3 --port 8000 --host localhost
```
To see all the available options `llama.cpp` provides, run:
```bash
symserver --help
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