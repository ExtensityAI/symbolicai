# Quick Start Guide for SymbolicAI

This guide will help you get started with SymbolicAI, demonstrating basic usage and key features.

## Basic Usage

First, import the library:

```python
import symai as ai
```

### Creating and Manipulating Symbols

```python
# Create a Symbol
sym = ai.Symbol("Welcome to our tutorial.")

# Translate the Symbol
translated = sym.translate('German')
print(translated)  # Output: Willkommen zu unserem Tutorial.

# Ranking objects
import numpy as np
sym = ai.Symbol(np.array([1, 2, 3, 4, 5, 6, 7]))
ranked = sym.rank(measure='numerical', order='descending')
print(ranked)  # Output: ['7', '6', '5', '4', '3', '2', '1']
```

### Evaluating Expressions

```python
# Word analogy
result = ai.Symbol('King - Man + Women').expression()
print(result)  # Output: Queen

# Sentence manipulation
result = ai.Symbol('Hello my enemy') - 'enemy' + 'friend'
print(result)  # Output: Hello my friend

# Fuzzy comparison
sym = ai.Symbol('3.1415...')
print(sym == numpy.pi)  # Output: True
```

### Causal Reasoning

```python
result = ai.Symbol('The horn only sounds on Sundays.') & ai.Symbol('I hear the horn.')
print(result)  # Output: It is Sunday.
```

## Using Different Engines

### Symbolic Engine (WolframAlpha)

```python
from symai import Interface
expression = Interface('wolframalpha')
res = expression('x^2 + 2x + 1')
print(res)  # Output: x = -1
```

### Speech Engine (Whisper)

```python
speech = Interface('whisper')
res = speech('path/to/audio.mp3')
print(res)  # Output: Transcribed text
```

### OCR Engine

```python
ocr = Interface('ocr')
res = ocr('https://example.com/image.jpg')
print(res['all_text'])  # Output: Extracted text from image
```

### Search Engine

```python
search = Interface('serpapi')
res = search('Birthday of Barack Obama')
print(res)  # Output: August 4, 1961
```

## Advanced Features

### Stream Processing

```python
from symai.components import *

stream = Stream(Sequence(
    Clean(),
    Translate(),
    Outline(),
))

sym = Symbol('<some long text>')
res = Symbol(list(stream(sym)))

expr = Cluster()
expr(res)
```

### Creating Custom Operations

```python
class Demo(ai.Expression):
    @ai.zero_shot(prompt="Generate a random integer between 0 and 10.",
                  constraints=[
                      lambda x: x >= 0,
                      lambda x: x <= 10
                  ])
    def get_random_int(self) -> int:
        pass

demo = Demo()
random_int = demo.get_random_int()
print(random_int)  # Output: A random integer between 0 and 10
```

This quick start guide covers the basics of SymbolicAI. For more detailed information and advanced usage, please refer to the full documentation and example notebooks.