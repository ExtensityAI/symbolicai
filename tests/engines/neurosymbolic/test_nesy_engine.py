from pathlib import Path

import pytest
from anthropic import Stream
from anthropic.types.message import Message
from google.genai import types  # Import for Gemini types
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.responses import Response

from symai import Expression, Symbol
from symai.backend.mixin.openai import SUPPORTED_REASONING_MODELS
from symai.backend.settings import SYMAI_CONFIG
from symai.components import Function
from symai.core_ext import bind

NEUROSYMBOLIC = SYMAI_CONFIG.get('NEUROSYMBOLIC_ENGINE_MODEL')
CLAUDE_THINKING = {"budget_tokens": 1024}
GEMINI_THINKING = {"thinking_budget": 1024}
CLAUDE_MAX_TOKENS = 4092
IS_RESPONSES_API = NEUROSYMBOLIC.startswith('responses:')

@bind(engine='neurosymbolic', property='compute_required_tokens')(lambda: 0)
def _compute_required_tokens(): pass

@pytest.mark.mandatory
def test_init():
    x = Symbol('This is a test!')

    if all(id not in NEUROSYMBOLIC for id in ['3-7', '4-0', '4-1', '4-5']):
        x.query('What is this?')
    else:
        x.query('What is this?', max_tokens=CLAUDE_MAX_TOKENS, thinking=CLAUDE_THINKING, raw_output=True)

    # if no errors are raised, then the test is successful
    assert True

@pytest.mark.mandatory
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('groq'), reason='feature not yet implemented')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('cerebras'), reason='feature not yet implemented')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('llama'), reason='feature not yet implemented')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('huggingface'), reason='feature not yet implemented')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('o3-mini'), reason='feature not supported by the model')
def test_vision():
    file = Path(__file__).parent.parent.parent.parent / 'assets' / 'images' / 'cat.jpg'
    x = Symbol(f'<<vision:{file}:>>')
    if all(id not in NEUROSYMBOLIC for id in ['3-7', '4-0', '4-1', '4-5']):
        res = x.query('What is in the image?')
    else:
        res = x.query('What is in the image?', max_tokens=CLAUDE_MAX_TOKENS, thinking=CLAUDE_THINKING)

    # it makes sense here to explicitly check if there is a cat; we are testing the vision component
    assert 'cat' in res.value

    file = 'https://raw.githubusercontent.com/ExtensityAI/symbolicai/main/assets/images/cat.jpg'
    x = Symbol(f'<<vision:{file}:>>')

    if all(id not in NEUROSYMBOLIC for id in ['3-7', '4-0', '4-1', '4-5']):
        res = x.query('What is in the image?')
    else:
        res = x.query('What is in the image?', max_tokens=CLAUDE_MAX_TOKENS, thinking=CLAUDE_THINKING)

    # same check but for url
    assert 'cat' in res.value

@pytest.mark.mandatory
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('groq'), reason='groq tokens computation is not supported')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('cerebras'), reason='cerebras tokens computation is not supported')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('llama'), reason='llamacpp tokens computation is not yet implemented')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('huggingface'), reason='huggingface tokens computation is not yet implemented')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('deepseek'), reason='deepseek tokens computation is not yet implemented')
def test_tokenizer():
    if NEUROSYMBOLIC.startswith('gemini'):
        messages = [
            {"role": "system", "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."},
            {"role": "user", "content": "New synergies will help drive top-line growth."},
            {"role": "model", "content": "Things working well together will increase revenue."},
            {"role": "user", "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
            {"role": "model", "content": "Let's talk later when we're less busy about how to do better."},
            {"role": "user", "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."}
        ]
    elif NEUROSYMBOLIC.startswith('claude'):
        messages = [
            {"role": "system", "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."},
            {"role": "user", "content": "New synergies will help drive top-line growth."},
            {"role": "assistant", "content": "Things working well together will increase revenue."},
            {"role": "user", "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
            {"role": "assistant", "content": "Let's talk later when we're less busy about how to do better."},
            {"role": "user", "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."}
        ]
    elif IS_RESPONSES_API:
        base_model = NEUROSYMBOLIC.replace('responses:', '')
        admin_role = 'developer' if base_model in SUPPORTED_REASONING_MODELS else 'system'
        messages = [
            {"role": admin_role, "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."},
            {"role": "user", "content": "New synergies will help drive top-line growth."},
            {"role": "assistant", "content": "Things working well together will increase revenue."},
            {"role": "user", "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
            {"role": "assistant", "content": "Let's talk later when we're less busy about how to do better."},
            {"role": "user", "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."}
        ]
    else:
        is_reasoning = NEUROSYMBOLIC in SUPPORTED_REASONING_MODELS
        admin_role = 'developer' if is_reasoning else ('system' if NEUROSYMBOLIC.startswith('gpt') else 'developer')
        messages = [
            {"role": admin_role, "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."},
            {"role": admin_role, "name": "example_user", "content": "New synergies will help drive top-line growth."},
            {"role": admin_role, "name": "example_assistant", "content": "Things working well together will increase revenue."},
            {"role": admin_role, "name": "example_user", "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
            {"role": admin_role, "name": "example_assistant", "content": "Let's talk later when we're less busy about how to do better."},
            {"role": "user", "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."}
        ]

    response = Expression.prompt(messages, raw_output=True)

    if NEUROSYMBOLIC.startswith('gemini'):
        api_tokens = response.usage_metadata.prompt_token_count
    elif NEUROSYMBOLIC.startswith('claude'):
        api_tokens = response[0].message.usage.input_tokens
    elif IS_RESPONSES_API:
        api_tokens = response.usage.input_tokens
    else:
        api_tokens = response.usage.prompt_tokens

    tik_tokens = _compute_required_tokens(messages)

    assert api_tokens == tik_tokens

@pytest.mark.mandatory
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('huggingface'), reason='feature not yet implemented')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('o1'), reason='feature not supported by the model')
@pytest.mark.skipif(NEUROSYMBOLIC == "gpt-5.1-chat-latest" or NEUROSYMBOLIC == "gpt-5-chat-latest", reason='feature not supported by the model with v1/chat/completions')
def test_tool_usage():
    if IS_RESPONSES_API:
        tools = [{
            "type": "function",
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
        }]
        fn = Function("Analyze the input request and select the most appropriate function to call from the provided options.", tools=tools)
        res, metadata = fn("what's the temperature in Bogotá, Colombia", raw_output=True, return_metadata=True)
        assert 'function_call' in metadata
        assert metadata['function_call']['name'] == 'get_weather'
        assert 'location' in metadata['function_call']['arguments']
        assert 'bogotá, colombia' in metadata['function_call']['arguments']['location'].lower()
    elif NEUROSYMBOLIC.startswith('o3') or NEUROSYMBOLIC.startswith('o4') or NEUROSYMBOLIC.startswith('gpt'):
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
        res, metadata = fn("what's the temperature in Bogotá, Colombia", raw_output=True, return_metadata=True)
        assert 'function_call' in metadata
        assert metadata['function_call']['name'] == 'get_weather'
        assert 'location' in metadata['function_call']['arguments']
        assert 'bogotá, colombia' in metadata['function_call']['arguments']['location'].lower()
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
        res, metadata = fn("What's the S&P 500 at today", raw_output=True, max_tokens=CLAUDE_MAX_TOKENS, thinking=CLAUDE_THINKING, return_metadata=True)
        assert 'function_call' in metadata
        assert metadata['function_call']['name'] == 'get_stock_price'
        assert 'ticker' in metadata['function_call']['arguments']
        assert 'spy' in metadata['function_call']['arguments']['ticker'].lower()
    elif NEUROSYMBOLIC.startswith('gemini'):
        # Test case 1: Callable Python function
        def get_capital(country: str | None = None) -> str:
            """Gets the capital city of a given country."""
            return "Paris"

        tools = [get_capital]
        fn = Function("Analyze the input request and select the most appropriate function to call from the provided options.", tools=tools)
        res, metadata = fn("What's the capital of France?", raw_output=True, thinking=GEMINI_THINKING, return_metadata=True)

        assert 'function_call' in metadata, 'Could fail if the function call is not recognized by the model; retry.'
        assert metadata['function_call']['name'] == 'get_capital'
        assert 'country' in metadata['function_call']['arguments']
        assert 'france' in metadata['function_call']['arguments']['country'].lower()

        # Test case 2: google.genai.types.Tool format
        fn_decl = types.FunctionDeclaration(
            name='get_current_weather',
            description='Get the current weather in a given location',
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'location': types.Schema(
                        type='STRING',
                        description='The city and state, e.g. San Francisco, CA',
                    ),
                },
                required=['location'],
            ),
        )

        tools = [fn_decl]
        fn = Function("Analyze the input request and select the most appropriate function to call from the provided options.", tools=tools)
        res, metadata = fn("What is the weather like in Boston?", raw_output=True, thinking=GEMINI_THINKING, return_metadata=True)

        assert 'function_call' in metadata
        assert metadata['function_call']['name'] == 'get_current_weather'
        assert 'location' in metadata['function_call']['arguments']
        assert 'boston' in metadata['function_call']['arguments']['location'].lower()

@pytest.mark.mandatory
def test_raw_output():
    if NEUROSYMBOLIC.startswith('claude'):
        if all(id not in NEUROSYMBOLIC for id in ['3-7', '4-0', '4-1', '4-5']):
            S = Expression.prompt('What is the capital of France?', raw_output=True)
        else:
            S = Expression.prompt('What is the capital of France?', raw_output=True, max_tokens=CLAUDE_MAX_TOKENS, thinking=CLAUDE_THINKING)
        # Accept list of events (when stream is consumed) or Message/Stream objects
        assert isinstance(S.value, (Message, Stream, list))
        if isinstance(S.value, list):
            # Verify it's a list of streaming events
            assert len(S.value) > 0
            # Check that list contains expected event types
            event_types = {type(event).__name__ for event in S.value}
            expected_types = {'RawMessageStartEvent', 'RawContentBlockStartEvent',
                            'RawContentBlockDeltaEvent', 'RawContentBlockStopEvent'}
            assert len(event_types & expected_types) > 0, f"Expected streaming events but got: {event_types}"
    elif IS_RESPONSES_API:
        S = Expression.prompt('What is the capital of France?', raw_output=True)
        assert isinstance(S.value, Response)
    elif NEUROSYMBOLIC.startswith('gpt') or NEUROSYMBOLIC.startswith('o1') or NEUROSYMBOLIC.startswith('o3'):
        S = Expression.prompt('What is the capital of France?', raw_output=True)
        assert isinstance(S.value, ChatCompletion)
    elif NEUROSYMBOLIC.startswith('gemini'):
        S = Expression.prompt('What is the capital of France?', raw_output=True)
        assert isinstance(S.value, types.GenerateContentResponse)

@pytest.mark.mandatory
def test_preview():
    preview_function = Function(
        "Return a JSON markdown string representation of the text, no matter what text is provided.",
        static_context="This is a static context.",
        dynamic_context="This is a dynamic context."
    )
    preview = preview_function("Hello, World!", preview=True)
    # assert that the output property `processed_input` is as expected
    assert preview.prop.processed_input == "Hello, World!"

@pytest.mark.skipif(NEUROSYMBOLIC.startswith('claude'), reason='Claude tokens computation is not yet implemented')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('gemini'), reason='Claude tokens computation is not yet implemented')
def test_token_truncator():
    file_path = (Path(__file__).parent.parent.parent / 'data/sample.txt').as_posix()
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
