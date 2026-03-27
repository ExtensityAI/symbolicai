# OCR Engine

Uses the [Mistral Document AI OCR API](https://docs.mistral.ai/capabilities/document_ai/basic_ocr)
for high-quality document and image text extraction with markdown output.

## Configuration

Set in your `symai.config.json`:

```json
{
    "OCR_ENGINE_API_KEY": "<MISTRAL_API_KEY>",
    "OCR_ENGINE_MODEL": "mistral-ocr-latest"
}
```

## Installation

```bash
pip install symbolicai[ocr]
```

## Usage

```python
from symai.interfaces import Interface

ocr = Interface('ocr')

# PDF document
res = ocr(document_url="https://arxiv.org/pdf/2201.04234")
print(res)  # assembled markdown

# Image
res = ocr(image_url="https://example.com/receipt.jpg")

# Per-page output
res = ocr(document_url="https://example.com/paper.pdf", per_page=True)
for page in res.value:
    print(page)

# Mistral-specific options
res = ocr(
    document_url="https://example.com/paper.pdf",
    table_format="markdown",        # "markdown", "html", or None
    extract_header=True,
    extract_footer=True,
)
```

## Images

Figures and charts detected in the document appear as placeholders in the markdown
(e.g. `![img-0.jpeg](img-0.jpeg)`). To extract the actual image data, pass
`include_image_base64=True`:

```python
res = ocr(document_url="https://arxiv.org/pdf/2201.04234", include_image_base64=True)

# markdown stays clean with placeholders
print(res)

# images available as a separate mapping: id -> base64 data URI
for img_id, data_uri in res.images.items():
    print(f"{img_id}: {len(data_uri)} chars")

# example: save an image to disk
import base64
data_uri = res.images["img-0.jpeg"]          # "data:image/jpeg;base64,/9j/4AAQ..."
header, b64 = data_uri.split(",", 1)
with open("img-0.jpeg", "wb") as f:
    f.write(base64.b64decode(b64))
```

## Supported formats

- **Documents** (`document_url`): PDF, PPTX, DOCX, and more
- **Images** (`image_url`): PNG, JPEG/JPG, AVIF, and more

## Usage tracking

Mistral OCR uses page-based billing. Use `MetadataTracker` to capture usage:

```python
from symai.components import MetadataTracker
from symai.interfaces import Interface
from symai.utils import RuntimeInfo

ocr = Interface('ocr')

with MetadataTracker() as tracker:
    res = ocr(document_url="https://arxiv.org/pdf/2201.04234")

# per-engine usage breakdown
usage_per_engine = RuntimeInfo.from_tracker(tracker, total_elapsed_time=0)
for (engine_name, model_name), info in usage_per_engine.items():
    print(f"{engine_name} ({model_name})")
    print(f"  calls: {info.total_calls}")
    print(f"  pages_processed: {info.extras.get('pages_processed', 0)}")
    print(f"  doc_size_bytes:  {info.extras.get('doc_size_bytes', 0)}")

# cost estimation
def ocr_pricing(info, cost_per_page=0.002):
    return info.extras.get("pages_processed", 0) * cost_per_page

total = RuntimeInfo(0, 0, 0, 0, 0, 0, 0, 0)
for _, info in usage_per_engine.items():
    total += RuntimeInfo.estimate_cost(info, ocr_pricing)

print(f"Estimated cost: ${total.cost_estimate:.4f}")
```
