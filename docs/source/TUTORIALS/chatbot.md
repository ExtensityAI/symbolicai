# Talking to Symbia 🤖

SymbiaChat is the built-in conversational agent that ships with SymbolicAI.
You don’t have to derive your own `ChatBot` subclass anymore—everything is
already wired together for you.

---

## Quick start

### 1 · From the terminal

```bash
symchat
```

### 2 · From Python

```python
from symai.chat import SymbiaChat

chat = SymbiaChat() # optional: verbose=True for debug prints
chat()
```

Both entry points start an interactive REPL. Type your messages and Symbia will
reply in real time.

---

## What can Symbia do?

Symbia continuously classifies your input and decides whether she can answer
directly, needs a tool, or should ask for clarification.
Below is the current toolbox with the *classification tags* it uses
internally (you will occasionally see them in debug mode):

| Tag | Engine | Typical prompt examples |
|-----|--------|-------------------------|
| `[WORLD-KNOWLEDGE]` | Internal LLM | “Who wrote *Dune*?” |
| `[SYMBOLIC]` | Symbolic math solver | “Integrate x² from 0 to 3.” |
| `[SEARCH]` | Web search (RAG) | “Give me the TL;DR of arXiv:2507.16075.” |
| `[SCRAPER]` | Website scraper | “Scrape all contributor names from <URL>.” |
| `[SPEECH-TO-TEXT]` | Audio transcription | “What does this file say? /path/to/audio.mp3” |
| `[TEXT-TO-IMAGE]` | Image generation | “Draw a friendly space cat.” |
| `[FILE]` | File reader | “Summarise /tmp/report.pdf.” |
| `[RECALL]` | Memory (short + long term) | “Remember that my dog’s name is Noodle.” |
| `[HELP]` | Capability list | “What can you do?” |
| `[DK]` | “Don’t know” ↔ ask user | Ambiguous or malformed input |
| `[EXIT]` | Quit session | “Goodbye”, “quit”, … |

---

## Memory model

1. **Short-term memory** – a fixed-size sliding window that keeps the last *n*
   conversation turns.
2. **Long-term memory** – a vector index stored on disk (`localdb/<index>.pkl`)
   for facts Symbia deems worth remembering.
   Ask “Do you remember …?” to trigger a `[RECALL]`.

Both memories are consulted automatically; you do not need to manage them.

---

## Under the hood (tl;dr)

```
ChatBot                # abstract base class
└── SymbiaChat         # concrete subclass you interact with
    ├─ In-context classification → picks capability tag
    ├─ SlidingWindowListMemory  → short-term memory
    ├─ Vector index interface   → long-term memory
    └─ self.interfaces[...]     → tooling shown above
```

If your use-case demands different behaviour, subclass `ChatBot` and override
`forward`, but for most applications `SymbiaChat` is already sufficient.

---

## Next steps

* Start a chat, explore the capabilities, and watch the classification tags in
  verbose mode for insight into Symbia’s decision making.

Happy chatting!
