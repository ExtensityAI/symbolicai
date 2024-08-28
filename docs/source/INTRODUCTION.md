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

## Future Directions

- Meta-learning semantic concepts on top of neuro-symbolic expressions
- Self-evolving and self-healing API
- Integration with reinforcement learning
- Advancement in prompt design and value alignment methods

SymbolicAI opens up new possibilities for creating applications that can perform self-analysis and self-repair, pushing the boundaries of what's possible with LLMs and neuro-symbolic computation.