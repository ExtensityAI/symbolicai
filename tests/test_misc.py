import json

import pytest

from symai import Expression, Symbol
from symai.backend.settings import SYMAI_CONFIG

NEUROSYMBOLIC = SYMAI_CONFIG.get('NEUROSYMBOLIC_ENGINE_MODEL')

@pytest.mark.skipif(NEUROSYMBOLIC.startswith('llama'), reason='llamacpp JSON format not yet supported')
def test_json_format():
    res = Expression.prompt(
        message=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": "Who won the world series in 2020? Return the answer in the following format: { 'team': 'str', 'year': 'int', 'coach': 'str'}}"}
      ],
        response_format={ "type": "json_object" },
        suppress_verbose_output=True
    )

    target = {'team': 'Los Angeles Dodgers', 'year': 2020, 'coach': 'Dave Roberts'}
    res = json.loads(res.value)
    assert res == target

def test_fill_format():
    text = '''
Oscar-winning actress Anne Hathaway has said that she had to kiss a series of actors during the audition process in the 2000s to “test for chemistry” on camera.

Now 41 and with dozens of movies to her name, Hathaway has revealed how early on in her career she went along with the “gross” request to “make out” with male actors to test their suitability for an unnamed movie.

{{fill}}

Speaking to V Magazine in an interview published Monday, Hathaway said: “Back in the 2000s — and this did happen to me — it was considered normal to ask an actor to make out with other actors to test for chemistry. Which is actually the worst way to do it.
'''

    S = Symbol(text)
    res = S.query("Fill in the text.", template_suffix="{{fill}}")
    assert "{{fill}}" not in S.value.replace("{{fill}}", res.value)

if __name__ == "__main__":
    pytest.main()
