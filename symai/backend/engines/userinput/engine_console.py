from typing import List

from ...base import Engine


class UserInputEngine(Engine):
    def __init__(self):
        super().__init__()

    def id(self) -> str:
        return 'userinput'

    def forward(self, argument):
        msg           = argument.prop.prepared_input
        kwargs        = argument.kwargs

        mock = kwargs['mock'] if 'mock' in kwargs else False
        if mock: # mock user input
            print(msg, end='') # print prompt
            rsp = mock
        else:
            rsp = input(msg)

        metadata = {}

        return [rsp], metadata

    def prepare(self, argument):
        # here the prompt marks the user input message
        argument.prop.prepared_input  = str(argument.prop.processed_input)
