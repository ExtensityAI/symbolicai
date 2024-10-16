# CLIP Engine

To perform text-based image few-shot classification, we use `CLIP`. This implementation is very experimental, and conceptually does not fully integrate the way we intend it, since the embeddings of CLIP and GPT-3 are not aligned (embeddings of the same word are not identical for both models). Aligning them remains an open problem for future research. For example, one could learn linear projections from one embedding space to the other.

The following example demonstrates how to classify the image of our generated cat from above and return the results as an array of probabilities:

```python
clip = Interface('clip')
res = clip('https://oaidalleapiprodscus.blob.core.windows.net/private/org-l6FsXDfth6...',
              ['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'])
```

```bash
:Output:
array([[9.72840726e-01, 6.34790864e-03, 2.59368378e-03, 3.41371237e-03,
        3.71197984e-03, 8.53193272e-03, 1.03346225e-04, 2.08464009e-03,
        1.77942711e-04, 1.94185617e-04]], dtype=float32)
```