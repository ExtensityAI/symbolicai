import json
import re
import sys
import time

import psutil
import pytest

from symai import Expression, Symbol
from symai.backend.settings import SYMAI_CONFIG
from symai.components import FileReader
from symai.formatter.formatter import CHUNK_REGEX
from symai.utils import format_bytes

NEUROSYMBOLIC = SYMAI_CONFIG.get('NEUROSYMBOLIC_ENGINE_MODEL')
CLAUDE_THINKING = {"budget_tokens": 1024}
CLAUDE_MAX_TOKENS = 4092 # Limit this, otherwise: "ValueError: Streaming is strongly recommended for operations that may take longer than 10 minutes."

@pytest.mark.mandatory
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('llama'), reason='llamacpp JSON format not yet supported')
@pytest.mark.skipif(NEUROSYMBOLIC.startswith('huggingface'), reason='huggingface JSON format not yet supported')
def test_json_format():
    admin_role = 'system' if \
        (NEUROSYMBOLIC.startswith('gpt') or
         NEUROSYMBOLIC.startswith('claude') or
         NEUROSYMBOLIC.startswith('gemini') or
         NEUROSYMBOLIC.startswith('groq')) \
         else 'developer'
    if all(id not in NEUROSYMBOLIC for id in ['3-7', '4-0']):
        res = Expression.prompt(
            message=[
                {"role": admin_role, "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": "Who won the world series in 2020? Return the answer in the following format: { 'team': 'str', 'year': 'int', 'coach': 'str'}}"}
        ],
            response_format={ "type": "json_object" },
            suppress_verbose_output=True
        )
    else:
        res = Expression.prompt(
            message=[
                {"role": admin_role, "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": "Who won the world series in 2020? Return the answer in the following format: { 'team': 'str', 'year': 'int', 'coach': 'str'}}"}
            ],
            response_format={ "type": "json_object" },
            suppress_verbose_output=True,
            max_tokens=CLAUDE_MAX_TOKENS,
            thinking=CLAUDE_THINKING
        )

    target = {'team': 'Los Angeles Dodgers', 'year': 2020, 'coach': 'Dave Roberts'}
    res = json.loads(res.value)
    assert res == target

@pytest.mark.mandatory
def test_fill_format():
    text = '''
Oscar-winning actress Anne Hathaway has said that she had to kiss a series of actors during the audition process in the 2000s to “test for chemistry” on camera.

Now 41 and with dozens of movies to her name, Hathaway has revealed how early on in her career she went along with the “gross” request to “make out” with male actors to test their suitability for an unnamed movie.

{{fill}}

Speaking to V Magazine in an interview published Monday, Hathaway said: “Back in the 2000s — and this did happen to me — it was considered normal to ask an actor to make out with other actors to test for chemistry. Which is actually the worst way to do it.
'''
    S = Symbol(text)
    if all(id not in NEUROSYMBOLIC for id in ['3-7', '4-0']):
        res = S.query("Fill in the text.", template_suffix="{{fill}}")
    else:
        res = S.query("Fill in the text.", template_suffix="{{fill}}", max_tokens=CLAUDE_MAX_TOKENS, thinking=CLAUDE_THINKING)
    assert "{{fill}}" not in S.value.replace("{{fill}}", res.value)

@pytest.mark.mandatory
def test_self_prompt():
    if all(id not in NEUROSYMBOLIC for id in ['3-7', '4-0']):
        sym = Symbol('np.log2(2)', self_prompt=True)
        res = Symbol('np.log2(2)').query('Is this equal to 1?', self_prompt=True)
    else:
        sym = Symbol('np.log2(2)', self_prompt=True, max_tokens=CLAUDE_MAX_TOKENS, thinking=CLAUDE_THINKING, suppress_verbose_output=True)
        res = Symbol('np.log2(2)').query('Is this equal to 1?', self_prompt=True, max_tokens=CLAUDE_MAX_TOKENS, thinking=CLAUDE_THINKING, suppress_verbose_output=True)
    assert any(s in res.value.lower() for s in ['yes', 'correct', 'true'])

def test_regex_format():
    # Test the regex pattern
    reader = FileReader()

    # Read from the arg[1] file
    test_text = reader('README.md')

    # Start measuring time and memory
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss

    # Apply the regex
    matches = CHUNK_REGEX.findall(str(test_text))

    # End measuring time and memory
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss

    # Calculate execution time and memory usage
    execution_time = end_time - start_time
    memory_used = end_memory - start_memory

    # Output results
    print(f"Number of chunks: {len(matches) if matches else 0}")
    print(f"Execution time: {execution_time:.3f} seconds")
    print(f"Memory used: {format_bytes(memory_used)}")

    # Output the first 100 matches (or fewer if there are less than 100)
    print('\nFirst 100 chunks:')
    if matches:
        for match in matches[:100]:
            print(repr(match[:50]))
    else:
        print('No chunks found.')

    # Output regex flags
    flags = []
    if CHUNK_REGEX.flags & re.MULTILINE:
        flags.append('m')
    if CHUNK_REGEX.flags & re.UNICODE:
        flags.append('u')
    print(f"\nRegex flags: {''.join(flags)}")

    # Check for potential issues
    if execution_time > 5:
        print('\nWarning: Execution time exceeded 5 seconds. The regex might be too complex or the input too large.')
    if memory_used > 100 * 1024 * 1024:
        print('\nWarning: Memory usage exceeded 100 MB. Consider processing the input in smaller chunks.')

if __name__ == "__main__":
    pytest.main()
