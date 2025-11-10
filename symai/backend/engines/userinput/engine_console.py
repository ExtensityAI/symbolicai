
from ....utils import UserMessage
from ...base import Engine


class UserInputEngine(Engine):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__

    def id(self) -> str:
        return 'userinput'

    def forward(self, argument):
        msg = argument.prop.prepared_input
        kwargs = argument.kwargs

        mock = kwargs.get('mock', False)
        if mock: # mock user input
            UserMessage(msg)
            rsp = mock
        else:
            rsp = input(msg)

        metadata = {}

        return [rsp], metadata

    def prepare(self, argument):
        # here the prompt marks the user input message
        argument.prop.prepared_input = str(argument.prop.processed_input)
