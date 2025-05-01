# Drawing Engine

We now expose two distinct high-level drawing interfaces:

1. **`gpt_image`** – a unified wrapper around OpenAI’s Images API (DALL·E 2/3 and `gpt-image-*`).
2. **`flux`** – Black Forest Labs’ Flux text-to-image models via api.us1.bfl.ai.

Both return a list of local PNG file paths.

---

## 1. OpenAI “gpt_image” Interface

Use `Interface('gpt_image')` to drive OpenAI’s Images API.
Supported models:
- `dall-e-2`
- `dall-e-3`
- `gpt-image-1`

### Example: Create

```python
from symai.interfaces import Interface

gpt_image = Interface('gpt_image')

paths = gpt_image(
    "a fluffy cat with a cowboy hat",
    operation="create",          # create | variation | edit
    model="dall-e-3",            # choose your model
    n=1,                          # number of images
    size=1024,                    # square size in px, or "1024x1024"
    response_format="url",        # "url" or "b64_json"
    # Extra for DALL·E-3:
    quality="standard",           # "standard" | "hd"
    style="vivid",                # "vivid" | "natural"
    # Extra for gpt-image-*:
    moderation="auto",            # "auto" | "strict"
    background="transparent",     # "auto" | "transparent"
    output_compression="png",     # "png" | "jpeg" | "webp"
    # if jpeg/webp you can also pass `output_compression=80` for quality
)

print(paths[0])  # → /tmp/tmpabcd.png
```

### Example: Variation

```python
from symai.interfaces import Interface
from pathlib import Path

gpt_image = Interface('gpt_image')

paths = gpt_image(
    operation="variation",
    model="dall-e-2",
    image_path=Path("assets/images/cat.png"),
    n=3,
    size=512,
    response_format="url",
)
```

### Example: Edit

```python
from symai.interfaces import Interface
from pathlib import Path

gpt_image = Interface('gpt_image')

paths = gpt_image(
    "Add medieval armor and scrolls in the background",
    operation="edit",
    model="gpt-image-1",
    image_path=Path("assets/images/cat.png"),
    mask_path=Path("assets/images/cat-mask.png"),  # optional
    n=1,
    size=512,
    quality="medium",   # only for gpt-image-*
)
```

### Supported Parameters

**Common** (all operations):

- `prompt` (str)
- `operation` (`"create"`|`"variation"`|`"edit"`)
- `model` (str)
- `n` (int, default=1)
- `size` (int or `"WxH"`)
- `response_format` (`"url"`|`"b64_json"`)

**Create-only**:

- DALL·E-3:
  - `quality` (`"standard"`|`"hd"`)
  - `style` (`"vivid"`|`"natural"`)
- gpt-image-*:
  - `quality` (`"auto"`|`"low"`|`"medium"`|`"high"`)
  - `moderation` (`"auto"`|`"strict"`)
  - `background` (`"auto"`|`"transparent"`)
  - `output_compression` (`"png"`|`"jpeg"`|`"webp"` or integer)

**Variation / Edit**:

- `image_path` (Path or str or list)
- `mask_path` (Path or str, edit only)

---

## 2. Black Forest Labs “flux” Interface

Use `Interface('flux')` to call Flux via https://api.us1.bfl.ai.
Supported models: any `flux-*`, e.g. configured in `SYMAI_CONFIG["DRAWING_ENGINE_MODEL"]`.

### Example

```python
from symai.interfaces import Interface

flux = Interface('flux')

paths = flux(
    "a futuristic city skyline at night",
    operation="create",        # currently only 'create' is implemented
    model="flux-pro-1.1",
    width=1024,                # default 1024
    height=768,                # default 768
    steps=50,                  # default 40
    guidance=7.5,              # default None
    seed=42,                   # default None
    safety_tolerance=2,        # default 2
    prompt_upsampling=False,   # default False
    interval=5,                # default None
    output_format="png",       # default 'png'
)

print(paths)  # → ['/tmp/tmp1234.png']
```

### Supported Parameters

- `model` (str)
- `width` (int, default 1024)
- `height` (int, default 768)
- `steps` (int, default 40)
- `guidance` (float)
- `seed` (int)
- `safety_tolerance` (int, default 2)
- `prompt_upsampling` (bool, default False)
- `interval` (int)
- `output_format` (str, default 'png')
- `except_remedy` (callable)

Under the hood Flux uses:

- POST `https://api.us1.bfl.ai/v1/{model}`
- GET  `https://api.us1.bfl.ai/v1/get_result?id={request_id}`

and writes out local PNG file(s).
