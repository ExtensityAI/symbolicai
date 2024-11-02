# Drawing Engine

To render images from text descriptions, we use `DALL·E 2 & 3`. The following example demonstrates how to draw a text description and return the image:

```python
from symai.interfaces import Interface


dall_e = Interface('dall_e')
res = dall_e('a fluffy cat with a cowboy hat', model='dall-e-3', image_size=1024)
```

```bash
:Output:
https://oaidalleapiprodscus.blob.core.windows...
```

Don't worry, we would never hide an image of a cat with a hat from you. Here is the image preview and [link](https://oaidalleapiprodscus.blob.core.windows.net/private/org-V7GXGSgpBiHOFLP9nrlYW6i5/user-YaUeHMUezl2Bxs7uFOX7FQTC/img-OIvTRtqat4ujvSksGsRC9Eae.png?st=2024-11-02T21%3A17%3A17Z&se=2024-11-02T23%3A17%3A17Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2024-11-02T01%3A40%3A53Z&ske=2024-11-03T01%3A40%3A53Z&sks=b&skv=2024-08-04&sig=Ta9INGFM6%2B5eWYQn7%2BbftInyjURQ2Z6Ew0hn4k9XBq0%3D):

<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/cat.png" width="200px">
