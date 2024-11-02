# Drawing Engine

To render images from text descriptions, we support multiple image generation models through different providers. Currently supported are `DALL·E 2 & 3` and Black Forest Labs' `Flux` models.

## DALL-E Example

```python
from symai.interfaces import Interface

# Initialize DALL-E interface
dall_e = Interface('dall_e')
res = dall_e('a fluffy cat with a cowboy hat', model='dall-e-3', image_size=1024)
```

```bash
:Output:
https://oaidalleapiprodscus.blob.core.windows...
```

## Flux Example

```python
from symai.interfaces import Interface

# Initialize Flux interface
flux = Interface('flux')
res = flux('a fluffy cat with a cowboy hat', model='flux-pro-1.1', width=1024, height=1024)
print(res)
```

## Supported Parameters

Both interfaces support passing all available API parameters through `kwargs`.

### DALL-E Parameters
```python
# All DALL-E API parameters are supported via kwargs:
# - n: number of images to generate
# - quality: "standard" or "hd"
# - style: "vivid" or "natural"
# - response_format: "url" or "b64_json"
# and more...
```

### Flux Parameters
```python
# Available Flux parameters:
# - model: model identifier (e.g., 'flux-pro-1.1')
# - width: image width in pixels (default: 1024)
# - height: image height in pixels (default: 768)
# - steps: number of inference steps (default: 40)
# - prompt_upsampling: whether to use prompt upsampling (default: False)
# - seed: random seed for reproducibility (default: None)
# - guidance: guidance scale (default: None)
# - safety_tolerance: safety check tolerance (default: 2)
# - interval: sampling interval (default: None)
# - output_format: output image format (default: 'png')
```

Don't worry, we would never hide an image of a cat with a hat from you! Here is the example Flux generated image preview:

<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/dev/assets/images/cat.png" width="512px">
