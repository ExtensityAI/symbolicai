import symai as ai
from pydoc import locate


class Interface(ai.Expression):
    def __new__(self, module, *args, **kwargs) :
        module = module.lower()
        module = module.replace('-', '_')
        self._module = module
        self.module_path = f'symai.extended.apis.{module}'
        return Interface.load_module_class(self.module_path, self._module)(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def load_module_class(module_path, class_name):
        module_ = locate(module_path)
        return getattr(module_, class_name)
