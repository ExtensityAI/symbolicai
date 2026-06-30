import traceback
from typing import Any


class PreProcessor:
    def __call__(self, argument) -> Any:
        raise NotImplementedError


class FormatPreProcessor(PreProcessor):
    def __init__(self, format: str) -> None:
        self.format_str = format

    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        return argument.args[0].format(**self.format_str)


class EqualsPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        a = str(argument.prop.instance)
        b = str(argument.args[0])
        return f"{a} == {b} =>"


class InterpretExpressionPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val = str(argument.prop.instance)
        return f"{val} =>"


class IndexPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        a = str(argument.prop.instance)
        b = str(argument.args[0])
        return f"{a} index {b} =>"


class SetIndexPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 2
        a = str(argument.prop.instance)
        b = str(argument.args[0])
        c = str(argument.args[1])
        return f"{a} index {b} set {c} =>"


class DeleteIndexPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        a = str(argument.prop.instance)
        b = str(argument.args[0])
        return f"{a} remove {b} =>"


class ComparePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        comp = argument.prop.operator
        a = str(argument.prop.instance)
        b = str(argument.args[0])
        return f"{a} {comp!s} {b} =>"


class RankPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) >= 1
        order = argument.prop.order
        measure = argument.prop.measure if argument.prop.measure else argument.args[0]
        list_ = str(argument.prop.instance)
        # convert to list if not already a list
        if "|" in list_ and "[" not in list_:
            list_ = [v.strip() for v in list_.split("|") if len(v.strip()) > 0]
            list_ = str(list_)
        return f"order: '{order!s}' measure: '{measure!s}' list: {list_} =>"


class ReplacePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 2
        return f"text '{argument.prop.instance}' replace '{argument.args[0]!s}' with '{argument.args[1]!s}'=>"


class IncludePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        return f"text '{argument.prop.instance}' include '{argument.args[0]!s}' =>"


class CombinePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        a = str(argument.prop.instance)
        b = str(argument.args[0])
        return f"{a} + {b} =>"


class NegatePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        stmt = str(argument.prop.instance)
        return f"{stmt} =>"


class ContainsPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        a = str(argument.prop.instance)
        b = str(argument.args[0])
        return f"{b} in {a} =>"


class StartsWithPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        a = str(argument.prop.instance)
        b = str(argument.args[0])
        return f"{a} startswith {b} =>"


class EndsWithPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        a = str(argument.prop.instance)
        b = str(argument.args[0])
        return f"{a} endswith {b} =>"


class IsInstanceOfPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        a = str(argument.prop.instance)
        b = str(argument.args[0])
        return f"{a} isinstanceof {b} =>"


class ExtractPatternPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        return f"from '{argument.prop.instance!s}' extract '{argument.args[0]!s}' =>"


class LogicExpressionPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) >= 1
        a = str(argument.prop.instance)
        b = str(argument.args[0])
        operator = argument.prop.operator
        return f"expr :{a}: {operator} :{b}: =>"


class SemanticMappingPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) >= 1
        topics = list(argument.prop.subscriber.keys())
        assert len(topics) > 0
        return f"topics {topics!s} in\ntext: '{argument.args[0]!s}' =>"


class SimulateCodePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val = argument.args[0] if len(argument.args) > 0 else ""
        return f"code '{argument.prop.instance!s}' params '{val!s}' =>"


class GenerateCodePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f"description '{argument.prop.instance!s}' =>"


class TextToOutlinePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f"text '{argument.prop.instance!s}' =>"


class UniquePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        unique = argument.prop.keys
        val = f"List of keys: {unique}\n"
        return f"{val}text '{argument.prop.instance!s}' =>"


class GenerateTextPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f"{argument.prop.instance!s}"


class ForEachPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val = str(argument.prop.instance)
        cond = argument.prop.condition
        apply = argument.prop.apply
        return f"{val} foreach '{cond}' apply '{apply}' =>"


class MapPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val = str(argument.prop.instance)
        context = argument.prop.context
        return f"{val} map '{context!s}' =>"


class ListPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val = str(argument.prop.instance)
        cond = argument.prop.condition
        return f"{val} list '{cond}' =>"


class QueryPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val = f"Data:\n{argument.prop.instance!s}\n"
        query = f"Context: {argument.prop.context}\n"
        return f"{val}{query}Answer:"


class SufficientInformationPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val = str(argument.prop.instance)
        query = str(argument.prop.query)
        return f"query {query} content {val} =>"


class ExpandFunctionPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val = str(argument.prop.instance)
        return f"{val} =>\ndef"


class ModifyPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        changes = argument.prop.changes
        return f"text '{argument.prop.instance!s}' modify '{changes!s}'=>"


class FilterPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        criteria = argument.prop.criteria
        include = "include" if argument.prop.include else "remove"
        return f"text '{argument.prop.instance!s}' {include} '{criteria!s}' =>"


class MapExpressionPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val = str(argument.prop.instance)
        instruction = argument.prop.context
        return f"text '{val}' {instruction} =>"


class ConsoleInputPreProcessor(PreProcessor):
    def __init__(self, skip: list[int] | None = None) -> None:
        super().__init__()
        skip = [skip] if skip and isinstance(skip, int) else skip
        self.skip = skip if skip is not None else []

    def __call__(self, argument) -> Any:
        return f"\n{argument.args[0]!s}\n$> "


class ConsolePreProcessor(PreProcessor):
    def __init__(self, skip: list[int] | None = None) -> None:
        super().__init__()
        skip = [skip] if skip and isinstance(skip, int) else skip
        self.skip = skip if skip is not None else []

    def __call__(self, argument) -> Any:
        # _func is called as: _func(self, self.value, *method_args, **method_kwargs)
        # argument.args[0] == symbol value, argument.prop.instance == Symbol object
        if argument.args:
            symbol_obj = argument.prop.instance
            symbol_value = argument.args[0]
            method_args = argument.args[1:]
            object_ = f"symbol_value: {symbol_value!r}"

            # kwargs passed at Symbol-construction time (e.g. test_kwarg=…)
            if symbol_obj._kwargs:
                object_ += f"\nsymbol_kwargs: {symbol_obj._kwargs}"

            if method_args:
                object_ += f"\nmethod_args: {method_args}"
            if argument.kwargs:
                object_ += f"\nmethod_kwargs: {argument.kwargs}"
        else:
            object_ = f"args: {argument.args}\nkwargs: {argument.kwargs}"
        return object_


class LanguagePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        language = argument.prop.language
        argument.prop.prompt = argument.prop.prompt.format(language)
        return f"{argument.prop.instance!s}"


class TextFormatPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        format_ = argument.prop.format
        argument.prop.prompt = argument.prop.prompt.format(format_)
        val = str(argument.prop.instance)
        return f"text {val} format '{format_}' =>"


class TranscriptionPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        modify_ = argument.prop.modify
        val = str(argument.prop.instance)
        return f"text {val} modify only '{modify_}' =>"


class StylePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        description = argument.prop.description
        text = f"[FORMAT]: {description}\n"
        libs = ", ".join(argument.prop.libraries)
        libraries = f"[LIBRARIES]: {libs}\n"
        content = f"[DATA]:\n{argument.prop.instance!s}\n\n"
        if argument.prop.template:
            placeholder = argument.prop.placeholder
            template = argument.prop.template
            parts = str(template).split(placeholder)
            assert len(parts) == 2, (
                f"Your template must contain exactly one placeholder '{placeholder}'  split:"
                + str(len(parts))
            )
            argument.prop.template_suffix = parts[1]
            return f'f"{text}{libraries}{content}"----------\n[TEMPLATE]:\n{parts[0]}'
        return f"{text}{libraries}{content}"


class UnwrapListSymbolsPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        entries = argument.prop.entries
        res = []
        # unwrap entries - pass through non-string types (bytes, Part, Content) for multimodal support
        for entry in entries:
            if isinstance(entry, (str, bytes)):
                res.append(entry)
            elif hasattr(entry, "__class__") and entry.__class__.__name__ in ("Part", "Content"):
                # Pass through Google genai types for multimodal embedding
                res.append(entry)
            else:
                res.append(str(entry))
        argument.prop.entries = res
        return ""


class ExceptionPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        ctxt = str(argument.prop.instance)
        ctxt = f"{ctxt}\n" if ctxt and len(ctxt) > 0 else ""
        val = argument.prop.query
        e = argument.prop.exception
        exception = "".join(traceback.format_exception_only(type(e), e)).strip()
        return f"context '{val}' exception '{exception}' code'{ctxt}' =>"


class CorrectionPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        ctxt = str(argument.prop.instance)
        ctxt = f"{ctxt}\n" if ctxt and len(ctxt) > 0 else ""
        val = argument.prop.context
        exception = ""
        if not argument.prop.exception:
            e = argument.prop.exception
            err_msg = "".join(traceback.format_exception_only(type(e), e)).strip()
            exception = f" exception '{err_msg}'"
        return f'context "{val}"{exception} code "{ctxt}" =>'


class EnumPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f"[{', '.join([str(x) for x in argument.prop.enum])}]\n"


class TextMessagePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f"Text: {argument.prop.instance!s}\n"


class SummaryPreProcessing(PreProcessor):
    def __call__(self, argument) -> Any:
        ctxt = f"Context: {argument.prop.context} " if argument.prop.context else ""
        return f"{ctxt}Text: {argument.prop.instance!s}\n"


class CleanTextMessagePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f"Text: '{argument.prop.instance!s}' =>"


class PredictionMessagePreProcessor(PreProcessor):
    def __call__(self, _argument) -> Any:
        return "Prediction:"


class ArrowMessagePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f"{argument.prop.instance!s} =>"


class ValuePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f"{argument.prop.instance!s}"
