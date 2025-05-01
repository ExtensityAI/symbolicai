# Indexing Engine

We use `Pinecone` to index and search for text. The following example demonstrates how to store text as an index and then retrieve the most related match:

```python
expr = Expression()
expr.add(Symbol('Hello World!').zip())
expr.add(Symbol('I like cookies!').zip())
res = expr.get(Symbol('hello').embedding, index_name='default_index').ast()
res['matches'][0]['metadata']['text'][0]
```

```bash
:Output:
Hello World!
```

Here, the `zip` method creates a pair of strings and embedding vectors, which are then added to the index. The line with `get` retrieves the original source based on the vector value of `hello` and uses `ast` to cast the value to a dictionary.

You can set several optional arguments for the indexing engine. For more details, see the `symai/backend/engine_pinecone.py` file.