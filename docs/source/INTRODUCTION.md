# Introduction

**SymbolicAI: A Neuro-Symbolic Perspective on Large Language Models (LLMs)**

## Overview

SymbolicAI is a framework that leverages machine learning, specifically Large Language Models (LLMs), as its foundation and composes operations based on task-specific prompting. Our approach adopts a divide-and-conquer strategy to break down complex problems into smaller, more manageable tasks. By reassembling these operations, we can solve intricate problems efficiently.

Read [**full paper here**](https://arxiv.org/abs/2402.00854).

## Key Features

- Seamless transition between differentiable and classical programming
- Neuro-symbolic computation using LLMs
- Composable operations for complex problem-solving
- Integration with various engines (OpenAI, WolframAlpha, OCR, etc.)
- Support for multimodal inputs and outputs

## 🤷‍♂️ Why SymbolicAI?

SymbolicAI aims to bridge the gap between classical programming (Software 1.0) and modern data-driven programming (Software 2.0). It enables the creation of software applications that harness the power of large language models while maintaining the benefits of composability and inheritance from object-oriented programming.

By using SymbolicAI, you can traverse the spectrum between the classical programming realm and the data-driven programming realm, as illustrated in the following figure:

<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/img5.png" width="720px">

We adopt a divide-and-conquer approach, breaking down complex problems into smaller, manageable tasks. We use the expressiveness and flexibility of LLMs to evaluate these sub-problems. By re-combining the results of these operations, we can solve the broader, more complex problem.

In time, and with sufficient data, we can gradually transition from general-purpose LLMs with `zero` and `few-shot` learning capabilities to specialized, fine-tuned models designed to solve specific problems. This strategy enables the design of operations with fine-tuned, task-specific behavior.

## Core Concepts

1. **Symbols**: All data objects are treated as symbols, with natural language as the primary interface for interaction.
2. **Operations**: Contextualized functions that manipulate symbols and return new objects.
3. **Expressions**: Non-terminal symbols that can be further evaluated, allowing for complex computational graphs.
4. **Engines**: Various backends (e.g., GPT-3, WolframAlpha, CLIP) that power different types of computations and transformations.

## <img src="https://media.giphy.com/media/mGcNjsfWAjY5AEZNw6/giphy.gif" width="50"> More fun facts!

SymbolicAI is fundamentally inspired by the [`neuro-symbolic programming paradigm`](https://arxiv.org/abs/2210.05050).

**Neuro-symbolic programming** is an artificial intelligence and cognitive computing paradigm that combines the strengths of deep neural networks and symbolic reasoning.

**Deep neural networks** are machine learning algorithms inspired by the structure and function of biological neural networks. They excel in tasks such as image recognition and natural language processing. However, they struggle with tasks that necessitate explicit reasoning, like long-term planning, problem-solving, and understanding causal relationships.

**Symbolic reasoning** uses formal languages and logical rules to represent knowledge, enabling tasks such as planning, problem-solving, and understanding causal relationships. While symbolic reasoning systems excel in tasks requiring explicit reasoning, they fall short in tasks demanding pattern recognition or generalization, like image recognition or natural language processing.

**Neuro-symbolic programming** aims to merge the strengths of both neural networks and symbolic reasoning, creating AI systems capable of handling various tasks. This combination is achieved by using neural networks to extract information from data and utilizing symbolic reasoning to make inferences and decisions based on that data. Another approach is for symbolic reasoning to guide the neural networks' generative process and increase interpretability.

**Embedded accelerators for LLMs** will likely be ubiquitous in future computation platforms, including wearables, smartphones, tablets, and notebooks. These devices will incorporate models similar to GPT-3, ChatGPT, OPT, or Bloom.

<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/img1.png" width="720px">

LLMs are expected to perform a wide range of computations, like natural language understanding and decision-making. Additionally, neuro-symbolic computation engines will learn how to tackle unseen tasks and resolve complex problems by querying various data sources for solutions and executing logical statements on top.
To ensure the content generated aligns with our objectives, it is crucial to develop methods for instructing, steering, and controlling the generative processes of machine learning models. As a result, our approach works to enable active and transparent flow control of these generative processes.

<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/img7.png" width="720px">

The figure above depicts this generative process as shifting the probability mass of an input stream toward an output stream in a contextualized manner. With properly designed conditions and expressions, you can validate and guide the behavior towards a desired outcome or repeat expressions that fail to meet requirements. Our approach consists of defining a set of _fuzzy_ operations to manipulate the data stream and condition LLMs to align with our goals. We regard all data objects – such as strings, letters, integers, and arrays – as symbols and view natural language as the primary interface for interaction. See the following figure:

<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/img10.png" width="720px">

As long as our goals can be expressed through natural language, LLMs can be used for neuro-symbolic computations.
Consequently, we develop operations that manipulate these symbols to construct new symbols. Each symbol can be interpreted as a statement, and multiple statements can be combined to formulate a logical expression.

By combining statements together, we can build causal relationship functions and complete computations, transcending reliance purely on inductive approaches. The resulting computational stack resembles a neuro-symbolic computation engine at its core, facilitating the creation of new applications in tandem with established frameworks.

## Future Directions

- Meta-learning semantic concepts on top of neuro-symbolic expressions
- Self-evolving and self-healing API
- Integration with reinforcement learning
- Advancement in prompt design and value alignment methods

SymbolicAI opens up new possibilities for creating applications that can perform self-analysis and self-repair, pushing the boundaries of what's possible with LLMs and neuro-symbolic computation.

## ⚡Limitations

We are constantly working to improve the framework and overcome limitations and issues. Just to name a few:

Engineering challenges:

* Our framework constantly evolves and receives bug fixes. However, we advise caution when considering it for production use cases. For example, the Stream class only estimates the prompt size by approximation, which can fail. One can also create more sophisticated prompt hierarchies and dynamically adjust the global context based on a state-based approach. This would allow for consistent predictions even for long text streams.
* Operations need further improvements, such as verification for biases, fairness, robustness, etc.
* The code may not be complete and is not yet optimized for speed and memory usage. It utilizes API-based LLMs due to limitations in computing resources.
* Code coverage is not yet complete, and we are still working on the documentation.
* Integrate with a more diverse set of models from [Hugging Face](https://huggingface.co/) or other platforms.
* Currently, we have not accounted for multi-threading and multi-processing.

Research challenges:

* To reliably use our framework, one needs to further explore how to fine-tune LLMs to specifically solve many of the proposed operations in a more robust and efficient manner.
* The experimental integration of CLIP aims to align image and text embeddings. Enabling decision-making of LLMs based on observations and performing symbolic operations on objects in images or videos would be a significant leap forward. This integration would work well with reinforcement learning approaches and enable us to control policies systematically (see also [GATO](https://www.deepmind.com/publications/a-generalist-agent)). Therefore, we need to train large multi-modal variants with image/video data and text data, describing scenes in high detail to obtain neuro-symbolic computation engines that can perform semantic operations similar to `move-towards-tree`, `open-door`, etc.
* Generalist LLMs are still highly over-parameterized, and hardware has not yet caught up to hosting these models on everyday machines. This limitation constrains the applicability of our approach not only on small data streams but also creates high latencies, reducing the amount of complexity and expressiveness we can achieve with our expressions.

## 👥 References, Related Work, and Credits

This project draws inspiration from the following works, among others:

* [Newell and Simon's Logic Theorist: Historical Background and Impact on Cognitive Modeling](https://www.researchgate.net/publication/276216226_Newell_and_Simon's_Logic_Theorist_Historical_Background_and_Impact_on_Cognitive_Modeling)
* [Search and Reasoning in Problem Solving](https://www.sciencedirect.com/science/article/abs/pii/S0004370283800034)
* [The Algebraic Theory of Context-Free Languages](http://www-igm.univ-mlv.fr/~berstel/Mps/Travaux/A/1963-7ChomskyAlgebraic.pdf)
* [Neural Networks and the Chomsky Hierarchy](https://arxiv.org/abs/2207.02098)
* [Binding Language Models in Symbolic Languages](https://arxiv.org/abs/2210.02875)
* [Tracr: Compiled Transformers as a Laboratory for Interpretability](https://arxiv.org/abs/2301.05062)
* [How can computers get common sense?](https://www.science.org/doi/10.1126/science.217.4566.1237)
* [Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu/)
* [SymPy: symbolic computing in Python](https://github.com/sympy/sympy)
* [Neuro-symbolic programming](https://arxiv.org/abs/2210.05050)
* [Fuzzy Sets](https://web.archive.org/web/20150813153834/http://www.cs.berkeley.edu/~zadeh/papers/Fuzzy%20Sets-Information%20and%20Control-1965.pdf)
* [An early approach toward graded identity and graded membership in set theory](https://www.sciencedirect.com/science/article/abs/pii/S0165011409005326?via%3Dihub)
* [From Statistical to Causal Learning](https://arxiv.org/abs/2204.00607)
* [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
* [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)
* [Aligning Language Models to Follow Instructions](https://openai.com/blog/instruction-following/)
* [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
* [Measuring and Narrowing the Compositionality Gap in Language Models](https://ofir.io/self-ask.pdf)
* [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)
* [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/abs/2107.13586)
* [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
* [Understanding Stereotypes in Language Models: Towards Robust Measurement and Zero-Shot Debiasing](https://arxiv.org/abs/2212.10678)
* [Connectionism and Cognitive Architecture: A Critical Analysis](https://ruccs.rutgers.edu/images/personal-zenon-pylyshyn/proseminars/Proseminar13/ConnectionistArchitecture.pdf)
* [Unit Testing for Concepts in Neural Networks](https://arxiv.org/abs/2208.10244)
* [Teaching Algorithmic Reasoning via In-context Learning](https://arxiv.org/abs/2211.09066)
* [PromptChainer: Chaining Large Language Model Prompts through Visual Programming](https://arxiv.org/abs/2203.06566)
* [Prompting Is Programming: A Query Language For Large Language Models](https://arxiv.org/abs/2212.06094)
* [Self-Instruct: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560)
* [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)
* [Wolfram|Alpha as the Way to Bring Computational Knowledge Superpowers to ChatGPT](https://writings.stephenwolfram.com/2023/01/wolframalpha-as-the-way-to-bring-computational-knowledge-superpowers-to-chatgpt/)
* [Build a GitHub support bot with GPT3, LangChain, and Python](https://dagster.io/blog/chatgpt-langchain)

### Comparison to Other Frameworks

Here is a brief list contrasting our approach with other frameworks:

* We focus on cognitive science and cognitive architectures research. We believe that the current state of the art in LLMs is not yet ready for general-purpose tasks. So, we concentrate on advances in concept learning, reasoning, and flow control of the generative process.
* We consider LLMs as one type of neuro-symbolic computation engine, which could take various shapes or forms, such as knowledge graphs, rule-based systems, etc. Hence, our approach is not necessarily limited to Transformers or LLMs.
* We aim to advance the development of programming languages and new programming paradigms, along with their programming stack, including neuro-symbolic design patterns that integrate with operators, inheritance, polymorphism, compositionality, etc. Classical object-oriented and compositional design patterns have been well-studied in the literature, but we offer a novel perspective on how LLMs integrate and augment fuzzy logic and neuro-symbolic computation.
* Our proposed prompt design helps combine object-oriented paradigms with machine learning models. We believe that prompt misalignments in their current form will be alleviated with further advances in Reinforcement Learning from Human Feedback and other value alignment methods. As a result, these approaches will address the need for prompt engineering or the ability to prompt hack statements, leading to much shorter zero- or few-shot examples (at least for small enough tasks). We envision the power of a divide-and-conquer approach by performing basic operations and recombining them to tackle complex tasks.
* We view operators/methods as being able to move along a spectrum between prompting and fine-tuning, based on task-specific requirements and data availability. We believe this approach is more general compared to prompting frameworks.
* We propose a general method for handling large context sizes and transforming a data stream problem into a search problem, related to **reasoning as a search problem** in [Search and Reasoning in Problem Solving](https://www.sciencedirect.com/science/article/abs/pii/S0004370283800034).

We hope that our work can be seen as complementary and offer a future outlook on how we would like to use machine learning models as an integral part of programming languages and their entire computational stack.