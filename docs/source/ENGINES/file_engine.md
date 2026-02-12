# File Engine

The file engine reads documents and converts them to text. Plain text files
(.txt, .md, .py, .json, etc.) are read directly via native Python I/O. Rich
formats (PDF, DOCX, PPTX, XLSX, HTML, EPUB, etc.) are converted to Markdown
via [markitdown](https://github.com/microsoft/markitdown).

## Installation

Plain text files work out of the box. For rich format support:

```bash
pip install 'symbolicai[files]'
```

## Reading a Single File

Use `Symbol.open()` to read any supported file into a `Symbol`:

```python
from symai import Symbol

# Plain text files use the standard backend (default)
text = Symbol('./README.md').open()
print(text.value)  # file contents as a string

# Or pass the path as an argument
text = Symbol().open('./README.md')
```

### Backends

The engine has two backends:

- **`standard`** (default) -- reads plain text formats via native Python I/O
- **`markitdown`** -- converts any supported format to Markdown via markitdown

Rich formats (PDF, DOCX, images, etc.) require the markitdown backend:

```python
# Rich formats require backend='markitdown'
pdf = Symbol('./paper.pdf').open(backend='markitdown')
doc = Symbol('./report.docx').open(backend='markitdown')

# Plain text formats also work with markitdown (e.g. CSV -> markdown table)
raw = Symbol('./data.csv').open()
# "name,age\nAlice,30\nBob,25"

md = Symbol('./data.csv').open(backend='markitdown')
# "| name | age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |"
```

If you try to read a rich format with the standard backend, the error message
will suggest switching to `backend='markitdown'`.

### Structured Parsing with `as_box`

For structured data formats (JSON, YAML, TOML, CSV), pass `as_box=True` to get
a [python-box](https://github.com/cdgriffith/Box) object with dot-access:

```python
# JSON / YAML / TOML → Box (dot-access dict)
config = Symbol('./config.json').open(as_box=True)
print(config.value.database.host)  # dot-access instead of ["database"]["host"]

# CSV → BoxList (list of Box rows)
rows = Symbol('./data.csv').open(as_box=True)
print(rows.value[0].name)  # first row's "name" column
```

Supported extensions for `as_box`: `.json`, `.yaml`, `.yml`, `.toml`, `.csv`.

## Batch Reading with FileReader

`FileReader` is an Expression component for reading multiple files at once.
It returns a `Symbol` whose value is a **list of strings** (one per file):

```python
from symai.components import FileReader

reader = FileReader()

# Single file -- still returns a list
result = reader('./README.md')
print(result.value)  # ["file contents..."]

# Multiple files (backend kwarg is forwarded to each file read)
result = reader(['./paper.pdf', './notes.md', './data.xlsx'], backend='markitdown')
print(len(result.value))  # 3
for content in result.value:
    print(content[:100])
```

### Parallel Reading

For large batches, pass `workers=N` to read files across multiple processes:

```python
reader = FileReader()
result = reader(file_paths, workers=4, backend='markitdown')
```

Each worker process initializes its own markitdown converter, so there is no
shared state to worry about.

### Discovering Files

`FileReader.get_files()` recursively lists supported files in a directory:

```python
files = FileReader.get_files('./documents/', max_depth=2)
# Returns list of paths with supported extensions

reader = FileReader()
result = reader(files, workers=4)
```

## Supported Formats

| Category | Extensions |
|----------|------------|
| Plain text (built-in) | .txt, .md, .py, .json, .yaml, .yml, .csv, .tsv, .toml, .xml, .log |
| Rich formats (markitdown) | .pdf, .docx, .pptx, .xlsx, .xls, .html, .htm, .epub, .ipynb, .zip |
| Images (markitdown) | .jpg, .jpeg, .png |
| Audio (markitdown) | .mp3, .wav, .m4a, .mp4 |

## LLM-Powered Features

When markitdown is installed, image files and PowerPoint slides can include
LLM-generated descriptions. This routes through SymAI's neurosymbolic engine
using your configured `NEUROSYMBOLIC_ENGINE_MODEL` and API key -- any vision-
capable backend works (OpenAI GPT-4o, Anthropic Claude, Google Gemini, etc.).
If the configured engine doesn't support vision, these converters still extract
metadata (EXIF) and text content without LLM descriptions.

### Image Captioning

Opening an image with `backend='markitdown'` sends it through the configured
vision model and returns a Markdown description:

```python
from symai import Symbol

caption = Symbol('./photo.png').open(backend='markitdown')
print(caption.value)
# "# Description:\n# Workspace with Laptops\n\nA warm-toned photograph
#  captures a modern workspace on a wooden desk..."
```

This works with `.jpg`, `.jpeg`, and `.png` files. The vision adapter
(`_SymaiVisionClient`) bridges markitdown's OpenAI-style API with SymAI's
engine pipeline, so captioning works regardless of which provider is configured
(OpenAI, Anthropic, Google, etc.).

For batch captioning across many images:

```python
from symai.components import FileReader

images = FileReader.get_files('./photos/', max_depth=1)
reader = FileReader()
captions = reader(images, workers=4, backend='markitdown')
for path, text in zip(images, captions.value):
    print(f"{path}: {text[:80]}...")
```
