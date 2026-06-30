import json

import pytest

from symai import Expression, Symbol
from symai.backend.settings import SYMAI_CONFIG

NEUROSYMBOLIC = SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_MODEL")
CLAUDE_THINKING = {"budget_tokens": 1024}
CLAUDE_MAX_TOKENS = 4092  # Limit this, otherwise: "ValueError: Streaming is strongly recommended for operations that may take longer than 10 minutes."
IS_RESPONSES_API = NEUROSYMBOLIC.startswith("responses:")


@pytest.mark.mandatory
@pytest.mark.skipif(
    NEUROSYMBOLIC.startswith("llama"), reason="llamacpp JSON format not yet supported"
)
@pytest.mark.skipif(NEUROSYMBOLIC.startswith("vllm"), reason="vllm JSON format not yet supported")
@pytest.mark.skipif(
    NEUROSYMBOLIC.startswith("huggingface"), reason="huggingface JSON format not yet supported"
)
def test_json_format():
    admin_role = (
        "system"
        if (
            NEUROSYMBOLIC.startswith("gpt")
            or NEUROSYMBOLIC.startswith("claude")
            or NEUROSYMBOLIC.startswith("gemini")
            or NEUROSYMBOLIC.startswith("groq")
            or NEUROSYMBOLIC.startswith("cerebras")
            or IS_RESPONSES_API
        )
        else "developer"
    )
    if all(id not in NEUROSYMBOLIC for id in ["3-7", "4-0", "4-1", "4-5"]):
        res = Expression.prompt(
            message=[
                {
                    "role": admin_role,
                    "content": "You are a helpful assistant designed to output JSON.",
                },
                {
                    "role": "user",
                    "content": "Who won the world series in 2020? Return the answer in the following format: { 'team': 'str', 'year': 'int', 'coach': 'str'}}",
                },
            ],
            response_format={"type": "json_object"},
            suppress_verbose_output=True,
        )
    else:
        res = Expression.prompt(
            message=[
                {
                    "role": admin_role,
                    "content": "You are a helpful assistant designed to output JSON.",
                },
                {
                    "role": "user",
                    "content": "Who won the world series in 2020? Return the answer in the following format: { 'team': 'str', 'year': 'int', 'coach': 'str'}}",
                },
            ],
            response_format={"type": "json_object"},
            suppress_verbose_output=True,
            max_tokens=CLAUDE_MAX_TOKENS,
            thinking=CLAUDE_THINKING,
        )

    target = {"team": "Los Angeles Dodgers", "year": 2020, "coach": "Dave Roberts"}
    res = json.loads(res.value)
    assert res == target


@pytest.mark.mandatory
def test_fill_format():
    text = """
Oscar-winning actress Anne Hathaway has said that she had to kiss a series of actors during the audition process in the 2000s to “test for chemistry” on camera.

Now 41 and with dozens of movies to her name, Hathaway has revealed how early on in her career she went along with the “gross” request to “make out” with male actors to test their suitability for an unnamed movie.

{{fill}}

Speaking to V Magazine in an interview published Monday, Hathaway said: “Back in the 2000s — and this did happen to me — it was considered normal to ask an actor to make out with other actors to test for chemistry. Which is actually the worst way to do it.
"""
    S = Symbol(text)
    if all(id not in NEUROSYMBOLIC for id in ["3-7", "4-0", "4-1", "4-5"]):
        res = S.query("Fill in the text.", template_suffix="{{fill}}")
    else:
        res = S.query(
            "Fill in the text.",
            template_suffix="{{fill}}",
            max_tokens=CLAUDE_MAX_TOKENS,
            thinking=CLAUDE_THINKING,
        )
    assert "{{fill}}" not in S.value.replace("{{fill}}", res.value)


@pytest.mark.mandatory
def test_self_prompt():
    if all(id not in NEUROSYMBOLIC for id in ["3-7", "4-0", "4-1", "4-5"]):
        sym = Symbol("np.log2(2)", self_prompt=True)
        res = Symbol("np.log2(2)").query("Is this equal to 1?", self_prompt=True)
    else:
        sym = Symbol(
            "np.log2(2)",
            self_prompt=True,
            max_tokens=CLAUDE_MAX_TOKENS,
            thinking=CLAUDE_THINKING,
            suppress_verbose_output=True,
        )
        res = Symbol("np.log2(2)").query(
            "Is this equal to 1?",
            self_prompt=True,
            max_tokens=CLAUDE_MAX_TOKENS,
            thinking=CLAUDE_THINKING,
            suppress_verbose_output=True,
        )
    assert any(s in res.value.lower() for s in ["yes", "correct", "true"])


if __name__ == "__main__":
    pytest.main()
