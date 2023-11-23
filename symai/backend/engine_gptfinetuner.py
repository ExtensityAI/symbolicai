import os
from typing import List

import openai
from openai.cli import FineTune

from ..utils import Args
from .base import Engine
from .settings import SYMAI_CONFIG


class GPTFineTuner(Engine):
    def __init__(self):
        super().__init__()
        config          = SYMAI_CONFIG
        openai.api_key  = config['NEUROSYMBOLIC_ENGINE_API_KEY']
        self.base_model = "babbage"

    def forward(self, *args, **kwargs) -> List[str]:
        assert '__cmd__' in kwargs, "Missing __cmd__ argument"
        rsp = None

        if kwargs['__cmd__'] == 'prepare_data':
            assert 'file' in kwargs, "Missing file argument"
            args = Args(file=kwargs['file'], quiet=True)
            FineTune.prepare_data(args)
            rsp = 'success'
        elif kwargs['__cmd__'] == 'create':
            assert 'train_file' in kwargs, "Missing `train_file` argument"
            base_model = kwargs['model'] if 'model' in kwargs else self.base_model
            if 'model' in kwargs:
                del kwargs['model']
            suffix     = kwargs['suffix'] if 'suffix' in kwargs else None # optional suffix name for the model
            if 'suffix' in kwargs:
                del kwargs['suffix']
            valid      = kwargs['valid_file'] if 'valid_file' in kwargs else None # optional validation file
            if 'valid_file' in kwargs:
                del kwargs['valid_file']
            n_epochs   = kwargs['n_epochs'] if 'n_epochs' in kwargs else 5
            if 'n_epochs' in kwargs:
                del kwargs['n_epochs']

            args = Args(check_if_files_exist=True,
                        training_file=kwargs['train_file'],
                        validation_file=valid,
                        model=base_model,
                        suffix=suffix,
                        n_epochs=n_epochs,
                        **kwargs)
            create_args = {
                "training_file": FineTune._get_or_upload(
                    args.training_file, args.check_if_files_exist
                ),
            }
            if args.validation_file:
                create_args["validation_file"] = FineTune._get_or_upload(
                    args.validation_file, args.check_if_files_exist
                )

            for hparam in (
                "model",
                "suffix",
                "n_epochs",
                # "batch_size",
                # "learning_rate_multiplier",
                # "prompt_loss_weight",
                # "compute_classification_metrics",
                # "classification_n_classes",
                # "classification_positive_class",
                # "classification_betas",
            ):
                attr = getattr(args, hparam)
                if attr is not None:
                    create_args[hparam] = attr

            rsp = openai.FineTune.create(**create_args)
        elif kwargs['__cmd__'] == 'delete':
            assert 'id' in kwargs, "Missing `id` argument"
            args = Args(id=kwargs['id'])
            rsp = openai.FineTune.delete(sid=args.id)
        elif kwargs['__cmd__'] == 'cancel':
            assert 'id' in kwargs, "Missing `id` argument"
            args = Args(id=kwargs['id'])
            rsp = openai.FineTune.cancel(id=args.id)
        elif kwargs['__cmd__'] == 'retrieve':
            assert 'id' in kwargs, "Missing `id` argument"
            args = Args(id=kwargs['id'])
            rsp = openai.FineTune.retrieve(id=args.id)
        elif kwargs['__cmd__'] == 'get':
            assert 'id' in kwargs, "Missing `id` argument"
            args = Args(id=kwargs['id'])
            rsp = openai.FineTune.retrieve(id=args.id)
        elif kwargs['__cmd__'] == 'list':
            rsp = openai.FineTune.list()
        elif kwargs['__cmd__'] == 'results':
            assert 'id' in kwargs, "Missing `id` argument"
            args = Args(id=kwargs['id'])
            fine_tune = openai.FineTune.retrieve(id=args.id)
            if "result_files" not in fine_tune or len(fine_tune["result_files"]) == 0:
                raise openai.error.InvalidRequestError(
                    f"No results file available for fine-tune {args.id}", "id"
                )
            result_file = openai.FineTune.retrieve(id=args.id)["result_files"][0]
            rsp = openai.File.download(id=result_file["id"])
        elif kwargs['__cmd__'] == 'collect':
            assert 'file' in kwargs, "Missing `file` argument"
            args = Args(file=kwargs['file'], key_value_pair=kwargs['key_value_pair'])
            # extract path from file path and create if not exists
            path = os.path.dirname(args.file)
            os.makedirs(path, exist_ok=True)
            # create file if not exists
            if not os.path.exists(args.file):
                with open(args.file, 'w') as f:
                    f.write(args.key_value_pair)
            else:
                with open(args.file, 'a') as f:
                    f.write(args.key_value_pair)
        else:
            raise ValueError(f"Invalid command: {kwargs['__cmd__']}")

        del kwargs['__cmd__']

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = args
            metadata['output'] = rsp
            metadata['model']  = self.base_model

        return [rsp], metadata

    def prepare(self, args, kwargs, wrp_params):
        pass
