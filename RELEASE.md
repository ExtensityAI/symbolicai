## 2026-03-28

**BREAKING CHANGES**
- OCR engine replaced: APILayer removed in favor of Mistral Document AI (`mistral-ocr-latest`)
- OCR interface now requires keyword args (`document_url=` / `image_url=`) instead of positional URL
- New config key `OCR_ENGINE_MODEL` required alongside `OCR_ENGINE_API_KEY`

### Added
- Mistral OCR engine with document and image support, per-page output, and image extraction
- `pip install symbolicai[ocr]` extra for the Mistral OCR dependency
- Support for `gpt-5.4-mini` and `gpt-5.4-nano` reasoning models
- Token tracking for `EmbeddingEngine` in `MetadataTracker`
- Page-based usage tracking for Mistral OCR in `MetadataTracker`
- `cache_control` kwarg override for Anthropic engines; pass `False` to disable

### Fixed
- Pin `chonkie` without extras to resolve `litellm` dependency conflict

### Removed
- APILayer OCR engine (`engine_apilayer.py`)
- Unused files: `app.py`, `build.py`, `icon_converter.py`, `installer.py`, `Dockerfile`, `docker-compose.yml`, `trusted_repos.yml`, `CITATION.cff`, `environment.yml`

### Security
- Bump `requests` to fix reported vulnerability
- Bump `pyasn1` to fix reported vulnerability
- Bump `pyjwt` to fix reported vulnerability
