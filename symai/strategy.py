import logging
from pydoc import locate
from pydantic import BaseModel

from .components import ValidatedFunction
from .symbol import Expression


NUM_REMEDY_RETRIES = 10


class BaseStrategy(ValidatedFunction):
    def __init__(self, data_model: BaseModel, *args, **kwargs):
        super().__init__(
            data_model=data_model,
            retry_count=NUM_REMEDY_RETRIES,
            **kwargs,
        )
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        # TODO: inherit the strategy
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def forward(self, *args, **kwargs):
        result, _ = super().forward(
            *args,
            payload=self.payload,
            template_suffix=self.template,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return result

    @property
    def payload(self):
        return None

    @property
    def static_context(self):
        raise NotImplementedError()

    @property
    def template(self):
        return "{{fill}}"


class Strategy(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def __new__(self, module: str, *args, **kwargs):
        self._module = module
        self.module_path = f'symai.extended.strategies'
        return Strategy.load_module_class(self.module_path, self._module)(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def load_module_class(module_path, class_name):
        module_ = locate(module_path)
        return getattr(module_, class_name)
