Engines
================================================================

Due to limited computing resources, we currently utilize OpenAI's GPT-3, ChatGPT and GPT-4 API for the neuro-symbolic engine. However, given adequate computing resources, it is feasible to use local machines to reduce latency and costs, with alternative engines like OPT or Bloom. This would enable recursive executions, loops, and more complex expressions.

Furthermore, we interpret all objects as symbols with different encodings and have integrated a set of useful engines that convert these objects into the natural language domain to perform our operations.

.. toctree::
   :maxdepth: 2
   
   symbolic_engine
   speech_engine
   ocr_engine
   search_engine
   webcrawler_engine
   drawing_engine
   file_engine
   indexing_engine
   clip_engine
   local_engine
   custom_engine


To read more on the different engines used by SymbolicAI:

* **Neuro-Symbolic Engine**: [OpenAI's LLMs (supported GPT-3, ChatGPT, GPT-4)](https://beta.openai.com/docs/introduction/overview)
  (as an experimental alternative using **llama.cpp** for local models)
* **Embedding Engine**: [OpenAI's Embedding API](https://beta.openai.com/docs/introduction/overview)
* **[Optional] Symbolic Engine**: [WolframAlpha](https://www.wolframalpha.com/)
* **[Optional] Search Engine**: [SerpApi](https://serpapi.com/)
* **[Optional] OCR Engine**: [APILayer](https://apilayer.com)
* **[Optional] SpeechToText Engine**: [OpenAI's Whisper](https://openai.com/blog/whisper/)
* **[Optional] WebCrawler Engine**: [Selenium](https://selenium-python.readthedocs.io/)
* **[Optional] Image Rendering Engine**: [DALL·E 2](https://openai.com/dall-e-2/)
* **[Optional] Indexing Engine**: [Pinecone](https://app.pinecone.io/)
* **[Optional] [CLIP](https://openai.com/blog/clip/) Engine**: 🤗 [Hugging Face](https://huggingface.co/) (experimental image and text embeddings)