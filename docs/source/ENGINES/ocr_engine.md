# OCR Engine

---

## ⚠️  Outdated or Deprecated Documentation ⚠️
This documentation is outdated and may not reflect the current state of the SymbolicAI library. This page might be revived or deleted entirely as we continue our development. We recommend using more modern tools that infer the documentation from the code itself, such as [DeepWiki](https://deepwiki.com/ExtensityAI/symbolicai). This will ensure you have the most accurate and up-to-date information and give you a better picture of the current state of the library.

---
To extract text from images, we can perform optical character recognition (OCR) with `APILayer`. The following example demonstrates how to transcribe an image and return the text:

```python
from symai.interfaces import Interface

ocr = Interface('ocr')
res = ocr('https://media-cdn.tripadvisor.com/media/photo-p/0f/da/22/3a/rechnung.jpg')
```

The OCR engine returns a dictionary with a key `all_text` where the full text is stored. For more details, refer to their documentation [here](https://apilayer.com/marketplace/image_to_text-api).
