import re

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_thinking(content: str) -> tuple[str | None, str]:
    """Split a `<think>...</think>` reasoning block out of raw model content.

    Matches the first `<think>...</think>` block, DOTALL so it spans newlines.
    Returns `(thinking, cleaned_content)` with both trimmed; `thinking` is `None`
    if the block is absent or empty. `content` is returned unchanged (not
    trimmed) when no block is found. Pure: `CerebrasClient` never applies this
    automatically, so callers can access raw content.
    """
    match = _THINK_BLOCK.search(content)
    if match is None:
        return None, content

    thinking = match.group(1).strip()
    cleaned = (content[: match.start()] + content[match.end() :]).strip()
    return thinking or None, cleaned
