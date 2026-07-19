# Custom Engine

If you want to replace or extend the functionality of our framework, you can do so by customizing the existing engines or creating new engines.
To create and use any other LLM as a backend you can for example change the `neurosymbolic` engine setting and register the new engine to the `EngineRepository`. The following example shows how to create a new `neurosymbolic` engine:

```python
from symai.backend.base import Engine
from symai.functional import EngineRepository

# setup an engine
class MyEngine(Engine):
  def id(self):
    return 'neurosymbolic'

  def prepare(self, argument):
    # get input from the pre-processors output and use *args, **kwargs and prop from argument
    # argument.prop contains all your kwargs accessible via dot `.` operation and additional meta info
    # such as function signature, system relevant info etc.
    prompts = argument.prop.processed_input
    args    = argument.args
    kwargs  = argument.kwargs
    # prepare the prompt statement as you want (take a look at the other engines like OpenAIEngine)
    ...
    # assign it to prepared_input; build_request consumes it from here
    argument.prop.prepared_input = ...

  def forward(self, argument):
    # get prep statement
    prompt = argument.prop.prepared_input
    # Your API / engine related call code here
    return ...

# register your engine
custom_engine = MyEngine()
EngineRepository.register('neurosymbolic', custom_engine, allow_engine_override=True)
```

Any engine is derived from the base class `Engine` and is then registered in the engines repository using its registry ID. The ID is for instance used in `core.py` decorators to address where to send the zero/few-shot statements using the class `EngineRepository`. You can find the `EngineRepository` defined in `functional.py` with the respective `query` method. Every engine has therefore three main methods you need to implement. The `id`, `prepare` and `forward` method. The `id` return the engine category. The `prepare` and `forward` methods have a signature variable called  `argument` which carries all necessary pipeline relevant data. For instance, the output of the `argument.prop.processed_input` contains the pre-processed output of the `PreProcessor` …

The built-in engines structure `forward` as three steps — `build_request(argument)` creates an `EngineAPIRequest` dataclass, `call_request(request)` executes it against the shared httpx transport in `symai/backend/transport.py` via `execute_engine_api_request` (which handles retries, timeouts and error mapping), and `parse_response(response)` validates the payload into a response model. The base `Engine` class declares all three and raises `NotImplementedError` for them, so you can adopt the same structure in your own engine.

If you don't want to re-write the entire engine code but overwrite the existing prompt `prepare` logic, you can do so by subclassing the existing engine and overriding the `prepare` method.

Here is an example of how to initialize your own engine. We will subclass the existing `OpenAIEngine` and override the `prepare` method. This method is called before the neural computation and can be used to modify the input prompt's parameters that will be passed in for execution; whatever you assign to `argument.prop.prepared_input` is what `build_request` turns into the API payload. In this example, we will replace the prompt with dummy text for illustration purposes:

```python
import os

from symai import Expression, Symbol
from symai.backend.engines.neurosymbolic.openai.engine import OpenAIEngine
from symai.functional import EngineRepository


class DummyEngine(OpenAIEngine):
    def __init__(self):
        super().__init__(model='gpt-4.1-mini', api_key=os.getenv('OPENAI_API_KEY', 'your-api-key-here'))

    def prepare(self, argument):
        argument.prop.prepared_input = [
            {'role': 'system', 'content': 'Write like Jack London!'},
            {'role': 'user', 'content': 'Go wild and generate something!'}
        ]

custom_engine = DummyEngine()
sym = Symbol()
EngineRepository.register('neurosymbolic', custom_engine, allow_engine_override=True)
res = sym.compose()
print(res)
```

If you want to build an engine from scratch against a raw REST endpoint instead of subclassing a provider engine, implement `build_request`, `call_request` and `parse_response` on top of the shared transport. This gives you retries, error mapping (`EngineAuthenticationError`, `EngineRateLimitError`, …) and timeouts for free, and lets you inject an `httpx.Client` (e.g. with a `MockTransport` for tests):

```python
import httpx

from symai.backend.base import Engine
from symai.backend.request import EngineAPIRequest, EngineRequestPayload
from symai.backend.transport import execute_engine_api_request


class EchoPayload(EngineRequestPayload):
    prompt: str


class LocalEchoEngine(Engine):
    def __init__(self, base_url: str, client: httpx.Client | None = None):
        super().__init__()
        self.base_url = base_url
        self.transport_client = client  # None falls back to the shared default client

    def id(self):
        return 'neurosymbolic'

    def prepare(self, argument):
        argument.prop.prepared_input = str(argument.prop.processed_input)

    def build_request(self, argument) -> EngineAPIRequest:
        return EngineAPIRequest(
            provider='local',
            operation='echo',
            payload=EchoPayload(prompt=argument.prop.prepared_input),
            method='POST',
            url=f'{self.base_url}/echo',
            headers={'Content-Type': 'application/json'},
        )

    def call_request(self, request):
        response = execute_engine_api_request(request, client=self.transport_client)
        return response.json()

    def parse_response(self, response):
        return response['echo']

    def forward(self, argument):
        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)


# exercise it without a live server by injecting a MockTransport
def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={'echo': request.read().decode()})


engine = LocalEchoEngine(
    base_url='http://localhost:8000',
    client=httpx.Client(transport=httpx.MockTransport(handler)),
)
```

To configure an engine, we can forward commands through `Expression` objects by using the `command` method. The `command` method passes on configurations (as `**kwargs`) to the engines and change functionalities or parameters. The functionalities depend on the respective engine.

In this example, we will enable `verbose` mode, where the engine will print out the methods it is executing and the parameters it is using. This is useful for debugging purposes:

```python
from symai import Expression

sym = Symbol('Hello World!')
Expression.command(engines=['neurosymbolic'], verbose=True)
res = sym.translate('German')
```

Finally, if you want to create a completely new engine but still maintain our workflow, you can use the `query` function from [`symai/functional.py`](https://github.com/ExtensityAI/symbolicai/blob/main/symai/functional.py) and pass in your engine along with all other specified objects (i.e., Prompt, PreProcessor, etc).
