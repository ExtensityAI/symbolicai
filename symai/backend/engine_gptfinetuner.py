from typing import List

from .base import Engine
from .settings import SYMAI_CONFIG
from ..utils import Args

import openai
from openai.cli import FineTune


class GPTFineTuner(Engine):
    def __init__(self):
        super().__init__()
        config          = SYMAI_CONFIG
        self.base_model = config['NEUROSYMBOLIC_ENGINE_MODEL']

    def forward(self, *args, **kwargs) -> List[str]:
        assert '__cmd__' in kwargs, "Missing __cmd__ argument"
        rsp = None

        if kwargs['__cmd__'] == 'prepare_data':
            assert 'file' in kwargs, "Missing file argument"
            args = Args(file=kwargs['file'], quiet=True)
            FineTune.prepare_data(args)
            rsp = 'success'
        elif kwargs['__cmd__'] == 'create':
            assert 'id' in kwargs, "Missing id argument"
            args = Args(id=kwargs['id'],
                        train=kwargs['train'],
                        model=kwargs['model'],
                        suffix=kwargs['suffix'])
            rsp = openai.FineTune.create(sid=args.id)
        elif kwargs['__cmd__'] == 'delete':
            assert 'id' in kwargs, "Missing id argument"
            args = Args(id=kwargs['id'])
            rsp = openai.FineTune.delete(sid=args.id)
        elif kwargs['__cmd__'] == 'cancel':
            assert 'id' in kwargs, "Missing id argument"
            args = Args(id=kwargs['id'])
            rsp = openai.FineTune.cancel(sid=args.id)
        elif kwargs['__cmd__'] == 'retrieve':
            assert 'id' in kwargs, "Missing id argument"
            args = Args(id=kwargs['id'])
            rsp = openai.FineTune.retrieve(id=args.id)
        elif kwargs['__cmd__'] == 'get':
            assert 'id' in kwargs, "Missing id argument"
            args = Args(id=kwargs['id'])
            rsp = openai.FineTune.retrieve(id=args.id)
        elif kwargs['__cmd__'] == 'list':
            rsp = openai.FineTune.list()
        elif kwargs['__cmd__'] == 'results':
            assert 'id' in kwargs, "Missing id argument"
            args = Args(id=kwargs['id'])
            fine_tune = openai.FineTune.retrieve(id=args.id)
            if "result_files" not in fine_tune or len(fine_tune["result_files"]) == 0:
                raise openai.error.InvalidRequestError(
                    f"No results file available for fine-tune {args.id}", "id"
                )
            result_file = openai.FineTune.retrieve(id=args.id)["result_files"][0]
            rsp = openai.File.download(id=result_file["id"])
        else:
            raise ValueError(f"Invalid command: {kwargs['__cmd__']}")

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = args
            metadata['output'] = rsp

        return [rsp], metadata

    def prepare(self, args, kwargs, wrp_params):
        pass
