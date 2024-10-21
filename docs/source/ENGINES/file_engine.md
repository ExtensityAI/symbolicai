# File Engine

To perform file operations, we use the operating system's file system. Currently, we support only PDF files and plain text files. This is an early stage, and we are working on more sophisticated file system access and remote storage. The following example demonstrates how to read a PDF file and return the text:

```python
expr = Expression()
res = expr.open('./LICENSE')
```

```bash
:Output:
BSD 3-Clause License\n\nCopyright (c) 2023 ...
```