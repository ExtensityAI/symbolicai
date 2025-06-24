import logging
from pydoc import locate

from symai import Expression


class Strategy(Expression):
    """A base class for implementing strategies."""

    def __init__(self, *args, **kwargs):
        """Initialize the strategy object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def __new__(self, module: str, *args, **kwargs):
        """Create a new instance of the strategy.

        Args:
            module (str): The module name of the strategy.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Strategy: An instance of the strategy.

        Raises:
            NotImplementedError: If __call__ method is not implemented.
        """
        module = module.lower()
        module = module.replace('-', '_')
        self._module = module
        self.module_path = f'symai.extended.strategies.{module}'
        return Strategy.load_module_class(self.module_path, self._module)(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Call the strategy.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: If not overridden by child class.
        """
        raise NotImplementedError()

    @staticmethod
    def load_module_class(module_path, class_name):
        """Load a module class dynamically.

        Args:
            module_path (str): The path of the module.
            class_name (str): The name of the class.

        Returns:
            class: The loaded module class.

        Raises:
            AttributeError: If the module or class is not found.
        """
        module_ = locate(module_path)
        return getattr(module_, class_name)