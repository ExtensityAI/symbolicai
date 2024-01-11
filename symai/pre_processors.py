import traceback
from typing import Any, List, Optional

from .utils import prep_as_str


class PreProcessor:
    def __call__(self, argument) -> Any:
        raise NotImplementedError()


class RawInputPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f"{str(argument.prop.prompt)}"


class JsonPreProcessor(PreProcessor):
    def __call__(self, argument) -> None:
        assert len(argument.args) == 1
        self.format = format
        value = str(argument.args[0])
        return f'{value} => [JSON_BEGIN]'


class FormatPreProcessor(PreProcessor):
    def __init__(self, format: str) -> None:
        self.format_str = format

    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        return argument.args[0].format(**self.format_str)


class EqualsPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        a = prep_as_str(argument.prop.instance)
        b = prep_as_str(argument.args[0])
        return f'{a} == {b} =>'


class InterpretExpressionPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) >= 1
        val = str(argument.args[0])
        val = val.replace('self', str(argument.prop.instance))
        return f"{val}"


class IndexPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        a = prep_as_str(argument.prop.instance)
        b = prep_as_str(argument.args[0])
        return f'{a} index {b} =>'


class SetIndexPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 2
        a = prep_as_str(argument.prop.instance)
        b = prep_as_str(argument.args[0])
        c = prep_as_str(argument.args[1])
        return f'{a} index {b} set {c} =>'


class DeleteIndexPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        a = prep_as_str(argument.prop.instance)
        b = prep_as_str(argument.args[0])
        return f'{a} remove {b} =>'


class PromptPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f"{argument.prop.prompt} $>"


class ComparePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        comp = argument.prop.operator
        a = prep_as_str(argument.prop.instance)
        b = prep_as_str(argument.args[0])
        return f"{a} {str(comp)} {b} =>"


class RankPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) >= 1
        order = argument.prop.order
        measure = argument.prop.measure if argument.prop.measure else argument.args[0]
        list_ = prep_as_str(argument.prop.instance)
        # convert to list if not already a list
        if '|' in list_ and not '[' in list_:
            list_ = [v.strip() for v in list_.split('|') if len(v.strip()) > 0]
            list_ = str(list_)
        return f"order: '{str(order)}' measure: '{str(measure)}' list: {list_} =>"


class ReplacePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 2
        return f"text '{argument.prop.instance}' replace '{str(argument.args[0])}' with '{str(argument.args[1])}'=>"


class IncludePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        return f"text '{argument.prop.instance}' include '{str(argument.args[0])}' =>"


class CombinePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        a = prep_as_str(argument.prop.instance)
        b = prep_as_str(argument.args[0])
        return f"{a} + {b} =>"


class TemplatePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        placeholder = argument.prop.placeholder
        template = argument.prop.template
        parts = str(template).split(placeholder)
        assert len(parts) == 2, f"Your template must contain exactly one placeholder '{placeholder}' split:" + str(len(parts))
        argument.prop.template_suffix = parts[1]
        return f'----------\n[Template]:\n{parts[0]}'


class NegatePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        stmt = prep_as_str(argument.prop.instance)
        return f"{stmt} =>"


class ContainsPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        a = prep_as_str(argument.prop.instance)
        b = prep_as_str(argument.args[0])
        return f"{b} in {a} =>"


class IsInstanceOfPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        a = prep_as_str(argument.prop.instance)
        b = prep_as_str(argument.args[0])
        return f"{a} isinstanceof {b} =>"


class ExtractPatternPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) == 1
        return f"from '{str(argument.prop.instance)}' extract '{str(argument.args[0])}' =>"


class SimpleSymbolicExpressionPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) >= 1
        val = str(argument.args[0])
        val = val.replace('self', str(argument.prop.instance))
        return f"expr :{val} =: =>"


class LogicExpressionPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) >= 1
        a = prep_as_str(argument.prop.instance)
        b = prep_as_str(argument.args[0])
        operator = argument.prop.operator
        return f"expr :{a}: {operator} :{b}: =>"


class SemanticMappingPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert len(argument.args) >= 1
        topics = list(argument.prop.subscriber.keys())
        assert len(topics) > 0
        return f"topics {str(topics)} in\ntext: '{str(argument.args[0])}' =>"


class SimulateCodePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val = argument.args[0] if len(argument.args) >= 0 else ''
        return f"code '{str(argument.prop.instance)}' params '{str(val)}' =>"


class GenerateCodePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f"description '{str(argument.prop.instance)}' =>"


class TextToOutlinePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f"text '{str(argument.prop.instance)}' =>"


class UniquePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        unique = argument.prop.keys
        val = f'List of keys: {unique}\n'
        return f"{val}text '{str(argument.prop.instance)}' =>"


class GenerateTextPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f"{str(argument.prop.instance)}"


class ClusterPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        assert isinstance(argument.prop.instance.value, list), "ClusterPreProcessor can only be applied to a list"
        return argument.prop.instance.value


class ForEachPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val   = prep_as_str(argument.prop.instance)
        cond  = argument.prop.condition
        apply = argument.prop.apply
        return f"{val} foreach '{cond}' apply '{apply}' =>"


class MapPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val     = prep_as_str(argument.prop.instance)
        context = argument.prop.context
        return f"{val} map '{str(context)}' =>"


class ListPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val  = prep_as_str(argument.prop.instance)
        cond = argument.prop.condition
        return f"{val} list '{cond}' =>"


class QueryPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val   = f'Data:\n{str(argument.prop.instance)}\n'
        query = f"Context: {argument.prop.context}\n"
        return f"{val}{query}Answer:"


class SufficientInformationPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val   = prep_as_str(argument.prop.instance)
        query = prep_as_str(argument.prop.query)
        return f'query {query} content {val} =>'


class ExpandFunctionPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        val = prep_as_str(argument.prop.instance)
        return f'{val} =>\ndef'


class ArgsPreProcessor(PreProcessor):
    def __init__(self, format: str = '') -> None:
        self.format = format

    def __call__(self, argument) -> Any:
        args_ = [str(arg) for arg in argument.args]
        args_ = [str(argument.prop.instance), *args_]
        return self.format.format(*args_)


class ModifyPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        changes = argument.prop.changes
        return f"text '{str(argument.prop.instance)}' modify '{str(changes)}'=>"


class FilterPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        criteria = argument.prop.criteria
        include = 'include' if argument.prop.include else 'remove'
        return f"text '{str(argument.prop.instance)}' {include} '{str(criteria)}' =>"


class ArgsToInputPreProcessor(PreProcessor):
    def __init__(self, skip: Optional[List[int]] = None) -> None:
        super().__init__()
        skip = [skip] if skip and isinstance(skip, int) else skip
        self.skip = skip if skip is not None else []

    def __call__(self, argument) -> Any:
        input_ = ''
        for i, arg in enumerate(argument.args):
            if i in self.skip:
                continue
            input_ += f"{str(arg)}\n"
        return input_


class SelfToInputPreProcessor(PreProcessor):
    def __init__(self, skip: Optional[List[int]] = None) -> None:
        super().__init__()
        skip = [skip] if skip and isinstance(skip, int) else skip
        self.skip = skip if skip is not None else []

    def __call__(self, argument) -> Any:
        input_ = f'{str(argument.prop.instance)}\n'
        return input_


class DataTemplatePreProcessor(PreProcessor):
    def __init__(self, skip: Optional[List[int]] = None) -> None:
        super().__init__()
        self.skip = skip if skip is not None else []

    def __call__(self, argument) -> Any:
        input_ = f'[Data]:\n{str(argument.prop.instance)}\n'
        for i, arg in enumerate(argument.args):
            if i in self.skip:
                continue
            input_ += f"{str(arg)}\n"
        return input_


class ConsoleInputPreProcessor(PreProcessor):
    def __init__(self, skip: Optional[List[int]] = None) -> None:
        super().__init__()
        skip = [skip] if skip and isinstance(skip, int) else skip
        self.skip = skip if skip is not None else []

    def __call__(self, argument) -> Any:
        input_ = f'\n{str(argument.args[0])}\n$> '
        return input_


class ConsolePreProcessor(PreProcessor):
    def __init__(self, skip: Optional[List[int]] = None) -> None:
        super().__init__()
        skip = [skip] if skip and isinstance(skip, int) else skip
        self.skip = skip if skip is not None else []

    def __call__(self, argument) -> Any:
        object_ = f"args: {argument.args}\nkwargs: {argument.kwargs}"
        return object_


class LanguagePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        language = argument.prop.language
        argument.prop.prompt = argument.prop.prompt.format(language)
        return f"{str(argument.prop.instance)}"


class TextFormatPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        format_ = argument.prop.format
        argument.prop.prompt = argument.prop.prompt.format(format_)
        val = prep_as_str(argument.prop.instance)
        return f"text {val} format '{format_}' =>"


class TranscriptionPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        modify_ = argument.prop.modify
        val = prep_as_str(argument.prop.instance)
        return f"text {val} modify only '{modify_}' =>"


class StylePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        description = argument.prop.description
        text = f'[FORMAT]: {description}\n'
        libs = ', '.join(argument.prop.libraries)
        libraries = f"[LIBRARIES]: {libs}\n"
        content = f'[DATA]:\n{str(argument.prop.instance)}\n\n'
        if argument.prop.template:
            placeholder = argument.prop.placeholder
            template    = argument.prop.template
            parts       = str(template).split(placeholder)
            assert len(parts) == 2, f"Your template must contain exactly one placeholder '{placeholder}'  split:" + str(len(parts))
            argument.prop.template_suffix = parts[1]
            return f'f"{text}{libraries}{content}"----------\n[TEMPLATE]:\n{parts[0]}'
        return f"{text}{libraries}{content}"


class UnwrapListSymbolsPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        entries = argument.prop.entries
        res = []
        # unwrap entries
        for entry in entries:
            if type(entry) is not str:
                res.append(str(entry))
            else:
                res.append(entry)
        argument.prop.entries = res
        return ""


class ExceptionPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        ctxt = prep_as_str(argument.prop.instance)
        ctxt = f"{ctxt}\n" if ctxt and len(ctxt) > 0 else ''
        val  = argument.prop.query
        e    = argument.prop.exception
        exception = "".join(traceback.format_exception_only(type(e), e)).strip()
        return f"context '{val}' exception '{exception}' code'{ctxt}' =>"


class CorrectionPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        ctxt = prep_as_str(argument.prop.instance)
        ctxt = f"{ctxt}\n" if ctxt and len(ctxt) > 0 else ''
        val  = argument.prop.context
        exception = ''
        if not argument.prop.exception:
            e         = argument.prop.exception
            err_msg   = "".join(traceback.format_exception_only(type(e), e)).strip()
            exception = f" exception '{err_msg}'"
        return f'context "{val}"{exception} code "{ctxt}" =>'


class EnumPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f'[{", ".join([str(x) for x in argument.prop.enum])}]\n'


class TextMessagePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f'Text: {str(argument.prop.instance)}\n'


class SummaryPreProcessing(PreProcessor):
    def __call__(self, argument) -> Any:
        ctxt = f"Context: {argument.prop.context} " if argument.prop.context else ''
        return f'{ctxt}Text: {str(argument.prop.instance)}\n'


class CleanTextMessagePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f"Text: '{str(argument.prop.instance)}' =>"


class CrawlPatternPreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return ''


class PredictionMessagePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f'Prediction:'


class ArrowMessagePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f'{str(argument.prop.instance)} =>'


class ValuePreProcessor(PreProcessor):
    def __call__(self, argument) -> Any:
        return f'{str(argument.prop.instance)}'

