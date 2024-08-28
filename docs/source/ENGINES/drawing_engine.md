# Drawing Engine

To render images from text descriptions, we use `DALL·E 2`. The following example demonstrates how to draw a text description and return the image:

```python
from symai.interfaces import Interface

dalle = Interface('dall_e')
res = dalle('a cat with a hat')
```

```bash
:Output:
https://oaidalleapiprodscus.blob.core.windows.net/private/org-l6FsXDfth6Uct ...
```

Don't worry, we would never hide an image of a cat with a hat from you. Here is the image preview and [link](https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/cat.jpg):

<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/cat.jpg" width="200px">