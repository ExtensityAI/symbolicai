from copy import deepcopy

from ....symbol import Result
from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG

try:
    import wolframalpha as wa
except ImportError:
    UserMessage("WolframAlpha is not installed. Please install it with `pip install symbolicai[wolframalpha]`", raise_with=ImportError)


class WolframResult(Result):
    def __init__(self, value) -> None:
        super().__init__(value)
        self.raw = value
        self._value = value


class WolframAlphaEngine(Engine):
    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        self.api_key = self.config['SYMBOLIC_ENGINE_API_KEY'] if api_key is None else api_key
        self.client = wa.Client(self.api_key)
        self.name = self.__class__.__name__

    def id(self) -> str:
        if self.config['SYMBOLIC_ENGINE_API_KEY']:
            return 'symbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'SYMBOLIC_ENGINE_API_KEY' in kwargs:
            self.api_key = kwargs['SYMBOLIC_ENGINE_API_KEY']
            self.client = wa.Client(self.api_key)

    def forward(self, argument):
        queries = argument.prop.prepared_input

        rsp = None
        try:
            rsp = self.client.query(queries)
            rsp = WolframResult(rsp)
        except Exception as e:
            UserMessage(f'Failed to interact with WolframAlpha: {e}.\n\n If you are getting an error related to "assert", that is a well-known issue with WolframAlpha. There is a manual fix for this issue: https://github.com/jaraco/wolframalpha/pull/34/commits/6eb3828ee812f65592e00629710fc027d40e7bd1', raise_with=ValueError)

        metadata = {}

        return [rsp], metadata

    def prepare(self, argument):
        argument.prop.prepared_input = str(argument.prop.processed_input)
