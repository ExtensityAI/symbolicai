# Vision

SymbolicAI allows image inputs to models that support vision. A URL or local image can be passed directly into a Symbol like so:

```python
# Supported formats: jpg, jpeg, png, webp
image = "https://example.com/image.png"
image_description = Symbol(f"Analyze this image <<vision:{image}:>> and describe it.").interpret()


image = "path/to/local/image.jpg"
image_description = Symbol(f"Analyze this image <<vision:{image}:>> and describe it.").interpret()


images = [
    "https://example.com/image1.png", 
    "https://example.com/image2.png",
    "https://example.com/image3.png"
]
rank = Symbol(f"Given the following images, please rank them in order of appeal."
            + f"Images: \n<<vision:{images[0]}:>>\n<<vision:{images[1]}:>>\n<<vision:{images[2]}:>>\n<<vision:{images[3]}:>>\n<<vision:{images[4]}:>>").interpret()
```
