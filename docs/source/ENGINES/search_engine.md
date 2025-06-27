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
    "SEARCH_ENGINE_MODEL": "sonar",
    …
}
```

Additionally, we can use OpenAI's search capabilities to answer queries and get citations for the information:

```python
from symai.interfaces import Interface
search = Interface("openai_search")
res = search('Who is Nicusor Dan?')
```

```bash
:Output:
"As of May 20, 2025, the acting president of Romania is Ilie Bolojan. He assumed this role on February 12, 2025, following the resignation of Klaus Iohannis. Bolojan, a member of the National Liberal Party, previously served as the president of the Senate and the mayor of Oradea. [1]\n\nRomania held a presidential election on May 18, 2025, in which Nicușor Dan, the current mayor of Bucharest, was elected as the new president. Dan, an independent candidate endorsed by several pro-European Union parties, won the election with 53.6% of the vote, defeating nationalist candidate George Simion. [2] He is expected to be inaugurated as president in the near future.\n\n\n## Nicușor Dan's Victory in Romanian Presidential Election:\n- [3]\n- [4]\n- [5] "
```

The OpenAI search engine returns a `SearchResult` object that includes citations for the information. You can access these citations using:

```python
citations = res.get_citations()
```

The engine supports various customization options, such as specifying user location and timezone:

```python
# Search with user location
res = search("What are popular tourist attractions nearby?",
             user_location={
                 "type": "approximate",
                 "country": "US",
                 "city": "New York",
                 "region": "New York"
             })

# Search with timezone
res = search("What local events are happening today?",
             user_location={
                 "type": "approximate",
                 "country": "JP",
                 "city": "Tokyo",
                 "region": "Tokyo",
                 "timezone": "Asia/Tokyo"
             })

# Control the amount of search context
res = search("Explain quantum computing developments", search_context_size="high")
```

Here's how to configure the OpenAI search engine:

```bash
{
    …
    "SEARCH_ENGINE_API_KEY": "sk-…",
    "SEARCH_ENGINE_MODEL": "gpt-4.1-mini",
    …
}
```
