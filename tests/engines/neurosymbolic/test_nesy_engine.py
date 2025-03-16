from pathlib import Path

from numpy import add
import pytest

from anthropic.types import ToolUseBlock
from symai import Expression, Symbol
from symai.components import Function
from symai.backend.settings import SYMAI_CONFIG
from symai.core_ext import bind
from openai.types.chat.chat_completion import ChatCompletion
from anthropic.types.message import Message


NEUROSYMBOLIC = SYMAI_CONFIG.get('NEUROSYMBOLIC_ENGINE_MODEL')

@bind(engine='neurosymbolic', property='compute_required_tokens')(lambda: 0)
def _compute_required_tokens(): pass

def test_init():
    x = Symbol('This is a test!')
    x.query('What is this?')

    # if no errors are raised, then the test is successful
    assert True

@pytest.mark.skipif(NEUROSYMBOLIC.startswith('llama'), reason='feature not yet implemented')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('huggingface'), reason='feature not yet implemented')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('o3-mini'), reason='feature not supported by the model')
def test_vision():
    file = Path(__file__).parent.parent.parent.parent / 'assets' / 'images' / 'cat.jpg'
    x = Symbol(f'<<vision:{file}:>>')
    res = x.query('What is in the image?')

    # it makes sense here to explicitly check if there is a cat; we are testing the vision component
    assert 'cat' in res.value

    file = 'https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/cat.jpg'
    x = Symbol(f'<<vision:{file}:>>')
    res = x.query('What is in the image?')

    # same check but for url
    assert 'cat' in res.value

@pytest.mark.skipif(NEUROSYMBOLIC.startswith('claude'), reason='Claude tokens computation is not yet implemented')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('llama'), reason='llamacpp tokens computation is not yet implemented')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('huggingface'), reason='huggingface tokens computation is not yet implemented')
def test_tokenizer():
    admin_role = 'system' if NEUROSYMBOLIC.startswith('gpt') else 'developer'
    messages = [
        {
            "role": admin_role,
            "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
        },
        {
            "role": admin_role,
            "name": "example_user",
            "content": "New synergies will help drive top-line growth.",
        },
        {
            "role": admin_role,
            "name": "example_assistant",
            "content": "Things working well together will increase revenue.",
        },
        {
            "role": admin_role,
            "name": "example_user",
            "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
        },
        {
            "role": admin_role,
            "name": "example_assistant",
            "content": "Let's talk later when we're less busy about how to do better.",
        },
        {
            "role": "user",
            "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
        },
    ]

    response = Expression.prompt(messages, raw_output=True)
    api_tokens = response.usage.prompt_tokens
    tik_tokens = _compute_required_tokens(messages)

    assert api_tokens == tik_tokens

@pytest.mark.skipif(NEUROSYMBOLIC.startswith('llama'), reason='feature not yet implemented')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('huggingface'), reason='feature not yet implemented')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('o1'), reason='feature not supported by the model')
def test_tool_usage():
    if NEUROSYMBOLIC.startswith('o3') or NEUROSYMBOLIC.startswith('gpt'):
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia"
                        }
                    },
                    "required": ["location"],
                }
            }
        }]
        fn = Function("Analyze the input request and select the most appropriate function to call from the provided options.", tools=tools)
        res = fn("what's the temperature in Bogotá, Colombia", raw_output=True)
        tool = None
        if res.choices[0].finish_reason == 'tool_calls':
            tool_name = res.choices[0].message.tool_calls[0].function.name
        assert tool_name == 'get_weather'
    elif NEUROSYMBOLIC.startswith('claude'):
        tools = [
          {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol.",
            "input_schema": {
              "type": "object",
              "properties": {
                "ticker": {
                  "type": "string",
                  "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
                }
              },
              "required": ["ticker"]
            }
          }
        ]
        fn = Function("Analyze the input request and select the most appropriate function to call from the provided options.", tools=tools)
        if '3-7' not in NEUROSYMBOLIC:
            res = fn("what's the temperature in Bogotá, Colombia", raw_output=True)
        else:
            thinking={"type": "enabled", "budget_tokens": 4092}
            res = fn("what's the S&P 500 at today", raw_output=True, thinking=thinking, max_tokens=16000)
        for block in res.content:
            if isinstance(block, ToolUseBlock):
                assert block.name == 'get_stock_price'

def test_raw_output():
    S = Expression.prompt('What is the capital of France?', raw_output=True)
    if NEUROSYMBOLIC.startswith('claude'):
        assert isinstance(S.value, Message)
    elif NEUROSYMBOLIC.startswith('gpt') or NEUROSYMBOLIC.startswith('o1') or NEUROSYMBOLIC.startswith('o3'):
        assert isinstance(S.value, ChatCompletion)

@pytest.mark.skipif(NEUROSYMBOLIC.startswith('claude'), reason='Claude tokens computation is not yet implemented')
def test_token_truncator():
    file_path = (Path(__file__).parent.parent.parent / 'data/pg1727.txt').as_posix()
    content = Symbol(file_path).open()
    admin_role = 'system' if NEUROSYMBOLIC.startswith('gpt') else 'developer'

    # Case 1; user exceeds
    _ = Expression.prompt([
        {'role': 'user', 'content': content.value},
        {'role': 'user', 'content': "What's the most tragic event in the novel?"}
    ])

    # Case 2; system exceeds
    _ = Expression.prompt([
        {'role': admin_role, 'content': content.value},
        {'role': 'user', 'content': "What's the most tragic event in the novel?"}
    ])

    # Case 3; both exceed
    _ = Expression.prompt([
        {'role': admin_role, 'content': content.value},
        {'role': 'user', 'content': content.value + "What's the most tragic event in the novel?"}
    ])

    # Try from Symbol too
    _ = content.query("What's the most tragic event in the novel?")

    assert True

if __name__ == '__main__':
    pytest.main()
