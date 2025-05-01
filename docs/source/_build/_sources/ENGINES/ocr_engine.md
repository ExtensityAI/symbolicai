# OCR Engine

To extract text from images, we can perform optical character recognition (OCR) with `APILayer`. The following example demonstrates how to transcribe an image and return the text:

```python
from symai.interfaces import Interface

ocr = Interface('ocr')
res = ocr('https://media-cdn.tripadvisor.com/media/photo-p/0f/da/22/3a/rechnung.jpg')
```

The OCR engine returns a dictionary with a key `all_text` where the full text is stored. For more details, refer to their documentation [here](https://apilayer.com/marketplace/image_to_text-api).

```bash
:Output:
China Restaurant\nMaixim,s\nSegeberger Chaussee 273\n22851 Norderstedt\nTelefon 040/529 16 2 ...
```