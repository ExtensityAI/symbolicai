import json
import logging
import time
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any

from symai.backend.usage import EngineUsageRecord

logger = logging.getLogger(__name__)

ENGINE_UNREGISTERED = "<UNREGISTERED/>"


class Engine(ABC):
    def __init__(
        self,
        *,
        client_timeout: float | None = None,
        client_max_retries: int | None = None,
    ) -> None:
        super().__init__()
        self.client_timeout = client_timeout
        self.client_max_retries = client_max_retries
        self.verbose = False
        self.logging = False
        self.log_level = logging.DEBUG
        self.time_clock = False
        # library code must not add handlers or touch the root logger; the
        # application owns logging configuration for the `symai` logger tree.
        self.logger = logger

    def _build_client_kwargs(self, base, *, timeout_key="timeout", max_retries_key="max_retries"):
        if self.client_timeout is not None:
            base[timeout_key] = self.client_timeout
        if self.client_max_retries is not None:
            base[max_retries_key] = self.client_max_retries
        return base

    def _validate_retry_params(self, retry_params):
        if not isinstance(retry_params, dict):
            msg = "retry_params must be a dictionary"
            raise ValueError(msg)

        keys = ["tries", "delay", "max_delay", "backoff", "jitter", "graceful"]
        if not all(key in retry_params for key in keys):
            msg = f"Available keys: {keys}"
            raise ValueError(msg)

        return retry_params

    def _validate_timeout_params(self, timeout_params):
        if not isinstance(timeout_params, dict):
            msg = "timeout_params must be a dictionary"
            raise ValueError(msg)

        keys = ["read", "connect"]
        if not all(key in timeout_params for key in keys):
            msg = f"Available keys: {keys}"
            raise ValueError(msg)

        return timeout_params

    def __call__(self, argument: Any) -> tuple[list[str], dict]:
        log = {"Input": {"self": self, "args": argument.args, **argument.kwargs}}
        start_time = time.time()

        self._trigger_input_handlers(argument)

        res, metadata = self.forward(argument)

        req_time = time.time() - start_time
        metadata["time"] = req_time
        if self.time_clock:
            print(f"{argument.prop.func}: {req_time} sec")
        log["Output"] = res
        if self.verbose:
            view = {k: v for k, v in list(log["Input"].items()) if k != "self"}
            input_ = f"{str(log['Input']['self'])[:50]}, {argument.prop.func!s}, {view!s}"
            print(f"{input_[:150]} {str(log['Output'])[:100]}")
        if self.logging:
            self.logger.log(self.log_level, log)

        self._trigger_output_handlers(argument, res, metadata)
        return res, metadata

    def _trigger_input_handlers(self, argument: Any) -> None:
        instance_metadata = getattr(argument.prop.instance, "_metadata", None)
        if instance_metadata is not None:
            input_handler = getattr(instance_metadata, "input_handler", None)
            if input_handler is not None:
                input_handler((argument.prop.processed_input, argument))
        argument_handler = argument.prop.input_handler
        if argument_handler is not None:
            argument_handler((argument.prop.processed_input, argument))

    def _trigger_output_handlers(self, argument: Any, result: Any, metadata: dict | None) -> None:
        instance_metadata = getattr(argument.prop.instance, "_metadata", None)
        if instance_metadata is not None:
            output_handler = getattr(instance_metadata, "output_handler", None)
            if output_handler:
                output_handler(result)
        argument_handler = argument.prop.output_handler
        if argument_handler:
            argument_handler((result, metadata))

    def id(self) -> str:
        return ENGINE_UNREGISTERED

    def preview(self, argument):
        # Used here to avoid backend.base <-> symbol circular import.
        from symai.symbol import (  # noqa
            Symbol,
        )

        class Preview(Symbol):
            def __repr__(self) -> str:
                """
                Get the representation of the Symbol object as a string.

                Returns:
                    str: The representation of the Symbol object.
                """
                return str(self.value.prop.prepared_input)

            def prepared_input(self):
                return self.value.prop.prepared_input

        return Preview(argument), {}

    def supported_request_kwargs(self) -> set[str]:
        return set()

    def usage_record_from_metadata(self, _metadata: dict) -> EngineUsageRecord | None:
        return None

    def ignored_request_kwargs(self) -> set[str]:
        from symai.core import Argument  # noqa: PLC0415

        return set(Argument.default_property_values())

    def collect_request_kwargs(
        self,
        argument: Any,
        supported_request_kwargs: set[str] | frozenset[str] | None = None,
    ) -> dict[str, Any]:
        supported = (
            set(supported_request_kwargs)
            if supported_request_kwargs is not None
            else self.supported_request_kwargs()
        )
        if argument.kwargs.get("strict_request_kwargs", False):
            allowed = supported | self.ignored_request_kwargs()
            unsupported = sorted(set(argument.kwargs) - allowed)
            if unsupported:
                msg = f"Unsupported request kwargs for {self.__class__.__name__}: {unsupported}"
                raise ValueError(msg)

        return {key: argument.kwargs[key] for key in supported if key in argument.kwargs}

    @classmethod
    def self_prompt(cls, existing_prompt: dict[str, str], **kwargs) -> dict[str, str]:
        from symai import core  # noqa: PLC0415
        from symai.backend.engines.neurosymbolic._prompts import (  # noqa: PLC0415
            prompt_registry,
        )

        messages = [
            {"role": "system", "content": prompt_registry.render("self_prompt")},
            {"role": "user", "content": json.dumps(existing_prompt, ensure_ascii=False)},
        ]

        @core.zero_shot(
            raw_input=True,
            processed_input=messages,
            response_format={"type": "json_object"},
            post_processors=[lambda response, _: json.loads(response)],
            **kwargs,
        )
        def generate_prompt(_instance):
            pass

        instance = SimpleNamespace(global_context=("", ""), _kwargs={})
        return generate_prompt(instance)

    def build_request(self, argument: Any) -> Any:
        msg = f'Method "build_request" not implemented for {self.__class__.__name__}.'
        raise NotImplementedError(msg)

    def call_request(self, request: Any) -> Any:
        msg = f'Method "call_request" not implemented for {self.__class__.__name__}.'
        raise NotImplementedError(msg)

    def parse_response(self, response: Any) -> tuple[list[str], dict[str, Any]]:
        msg = f'Method "parse_response" not implemented for {self.__class__.__name__}.'
        raise NotImplementedError(msg)

    @abstractmethod
    def forward(self, *args: Any, **kwds: Any) -> list[str]:
        raise NotADirectoryError

    @abstractmethod
    def prepare(self, argument):
        raise NotImplementedError

    def command(self, *_args, **kwargs):
        if kwargs.get("verbose"):
            self.verbose = kwargs["verbose"]
        if kwargs.get("logging"):
            self.logging = kwargs["logging"]
        if kwargs.get("log_level"):
            self.log_level = kwargs["log_level"]
        if kwargs.get("time_clock"):
            self.time_clock = kwargs["time_clock"]

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        """
        Get the representation of the Symbol object as a string.

        Returns:
            str: The representation of the Symbol object.
        """
        # class with full path
        class_ = self.__class__.__module__ + "." + self.__class__.__name__
        hex_ = hex(id(self))
        return f"<class {class_} at {hex_}>"
