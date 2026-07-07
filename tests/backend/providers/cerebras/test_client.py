from symai.backend.providers.cerebras.client import extract_thinking

# --- extract_thinking ---------------------------------------------------------


def test_extract_thinking_returns_trimmed_thinking_and_content():
    thinking, content = extract_thinking("<think>reasoning here</think>the answer")

    assert thinking == "reasoning here"
    assert content == "the answer"


def test_extract_thinking_captures_multiline_block_with_dotall():
    raw = "<think>line one\nline two\nline three</think>final answer"

    thinking, content = extract_thinking(raw)

    assert thinking == "line one\nline two\nline three"
    assert content == "final answer"


def test_extract_thinking_no_tags_returns_none_and_same_content():
    raw = "just a plain answer, no reasoning tags here"

    thinking, content = extract_thinking(raw)

    assert thinking is None
    assert content == raw


def test_extract_thinking_empty_block_returns_none_thinking_and_stripped_content():
    thinking, content = extract_thinking("<think></think>the answer")

    assert thinking is None
    assert content == "the answer"
