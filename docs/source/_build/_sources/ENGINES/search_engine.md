# Search Engine

To obtain fact-based content, we can perform search queries via `SerpApi` with a `Google` backend. The following example demonstrates how to search for a query and return the results:

```python
from symai.interfaces import Interface

search = Interface('serpapi')
res = search('Birthday of Barack Obama')
```

```bash
:Output:
August 4, 1961
```

Here's a quick example for how to set it up:
```bash
{
    …
    "SEARCH_ENGINE_API_KEY": …,
    "SEARCH_ENGINE_ENGINE": "google",
    …
}
```

We can also use PerplexityAI to search for a query and return the results:

```python
from symai.interfaces import Interface
search = Interface("perplexity")
res = search('What is a quantum computer?', system_message='You are Rick from Rick and Morty. You reply back as Rick would reply to Morty.')
```

```bash
:Output:
Morty, a quantum computer is like a super-duper, hyper-advanced version of the old computer I used to build in my garage. It uses the principles of quantum mechanics to process information in ways that classical computers can't even dream of.

Here's the deal: instead of using those boring old bits that are either 0 or 1, quantum computers use something called qubits. These qubits can exist in multiple states at the same time, thanks to this weird phenomenon called superposition. It's like when you flip a coi
n and it's both heads and tails until you look at it—same idea, Morty.

And then there's entanglement. This means that if you have two qubits, the state of one can depend on the state of the other, no matter how far apart they are. It's like having a secret handshake with your buddy that works even if you're on opposite sides of the multive
rse.

So, with these qubits and their superposition and entanglement, quantum computers can perform calculations at speeds that would make even the most powerful supercomputers look like they're running on a Commodore 64. They can simulate molecular interactions, break encryp
tion codes faster than you can say "Wubba lubba dub dub," and solve problems that are currently unsolvable for classical computers.

Now, if you'll excuse me, I have some real science to attend to. Don't get too excited, Morty; it's still just a bunch of fancy math and physics. But hey, it's cool stuff, right?
```

Please note that the system_message is optional and can be used to provide context to the model. For all available kwargs that can be passed to the perplexity engine, please refer to the PerplexityAI documentation. Also, please see the available supported engines for PerplexityAI here. Here's a quick example for how to set it up:

```bash
{
    …
    "SEARCH_ENGINE_API_KEY": "pplx-…",
    "SEARCH_ENGINE_MODEL": "llama-3.1-sonar-small-128k-online",
    …
}
```