import traceback
from typing import Any, List, Optional
from .utils import prep_as_str


class PreProcessor:
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError()
    
    def override_reserved_signature_keys(self, wrp_params, *args: Any, **kwds: Any) -> Any:
        def _override_reserved_signature_keys(key) -> Any:
            assert type(wrp_params[key]) == type(kwds[key]), "Your function signature uses reserved keys from the decorator, but you passed in a different type of value. This is not allowed due to disambiguation issues in the arguments. We recommend to align the types of reserved keys in your function signature or rename your keyword."
            if isinstance(kwds[key], dict):
                wrp_params[key].update(kwds[key])
                del kwds[key]
            elif isinstance(kwds[key], list):
                for item in kwds[key]:
                    if item not in wrp_params[key]:
                        wrp_params[key].append(item)
            else:
                wrp_params[key] = kwds[key]
                del kwds[key]
        # for the case that the signature has the key
        for key in list(wrp_params['signature'].parameters):
            if key in wrp_params:
                # validate that the reserved key is passed in as a keyword argument
                assert key in kwds, "Your function signature uses reserved keys from the decorator, but you did not pass them in as keyword arguments when calling the function. This is not allowed due to disambiguation issues in the arguments. We recommend to add reserved keys to the last position of your function signature and call them as keyword arguments."
                _override_reserved_signature_keys(key)
        # for the case that the signature does not have the key but the user uses anyways as a keyword argument, take the value
        for key in list(kwds.keys()):
            if key not in list(wrp_params['signature'].parameters):
                _override_reserved_signature_keys(key)


class EqualsPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) == 1
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        a = prep_as_str(wrp_self)
        b = prep_as_str(args[0])
        return f'{a} == {b} =>'
    

class WolframAlphaPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) >= 1
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        val = str(args[0])
        val = val.replace('self', str(wrp_self))
        return f"{val}"

    
class IndexPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) == 1
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        a = prep_as_str(wrp_self)
        b = prep_as_str(args[0])
        return f'{a} index {b} =>'
    
    
class SetIndexPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) == 2
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        a = prep_as_str(wrp_self)
        b = prep_as_str(args[0])
        c = prep_as_str(args[1])
        return f'{a} index {b} set {c} =>'
    
    
class DeleteIndexPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) == 1
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        a = prep_as_str(wrp_self)
        b = prep_as_str(args[0])
        return f'{a} remove {b} =>'

    
class PromptPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        wrp_params['prompt'] = f"{wrp_params['prompt']} $>"
        return f"{wrp_params['prompt']} $>"
    
    
class FlattenListExamplesPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        vals = ', '.join(wrp_params['examples'])
        return f'{vals}, '

    
class ComparePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) == 1
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        comp = wrp_params.get('operator', '>')
        a = prep_as_str(wrp_self)
        b = prep_as_str(args[0])
        return f"{a} {str(comp)} {b} =>"
    
    
class RankPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) >= 1
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        order = wrp_params.get('order', 'desc')
        measure = wrp_params['measure'] if 'measure' in wrp_params else args[0]
        list_ = prep_as_str(wrp_self)
        # convert to list if not already a list
        if '|' in list_ and not '[' in list_:
            list_ = [v.strip() for v in list_.split('|') if len(v.strip()) > 0]
            list_ = str(list_)
        return f"order: '{str(order)}' measure: '{str(measure)}' list: {list_} =>"


class ReplacePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) == 2
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return f"text '{wrp_self}' replace '{str(args[0])}' with '{str(args[1])}'=>"
    
    
class IncludePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) == 1
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return f"text '{wrp_self}' include '{str(args[0])}' =>"
    

class CombinePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) == 1
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        a = prep_as_str(wrp_self)
        b = prep_as_str(args[0])
        return f"{a} + {b} =>"
    
    
class TemplatePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        placeholder = wrp_params['placeholder']
        template = wrp_params['template']
        parts = str(template).split(placeholder)
        assert len(parts) == 2, f"Your template must contain exactly one placeholder '{placeholder}'"
        wrp_params['template_suffix'] = parts[1]
        return f'----------\n[Template]:\n{parts[0]}'
    

class NegatePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        stmt = prep_as_str(wrp_self)
        return f"{stmt} =>"
    
    
class ContainsPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) == 1
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        a = prep_as_str(wrp_self)
        b = prep_as_str(args[0])
        return f"{b} in {a} =>"
    
    
class IsInstanceOfPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) == 1
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        a = prep_as_str(wrp_self)
        b = prep_as_str(args[0])
        return f"{a} isinstanceof {b} =>"
    
    
class ExtractPatternPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) == 1
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return f"from '{str(wrp_self)}' extract '{str(args[0])}' =>"
    
    
class SimpleSymbolicExpressionPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) >= 1
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        val = str(args[0])
        val = val.replace('self', str(wrp_self))
        return f"expr :{val} =: =>"
    
    
class LogicExpressionPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) >= 1
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        a = prep_as_str(wrp_self)
        b = prep_as_str(args[0])
        operator = wrp_params['operator']
        return f"expr :{a}: {operator} :{b}: =>"


class SemanticMappingPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) >= 1
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        topics = list(wrp_params['subscriber'].keys())
        assert len(topics) > 0
        return f"topics {str(topics)} in\ntext: '{str(args[0])}' =>"


class SimulateCodePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        val = args[0] if len(args) >= 0 else ''
        return f"code '{str(wrp_self)}' params '{str(val)}' =>"
    
    
class GenerateCodePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return f"description '{str(wrp_self)}' =>"
    
    
class TextToOutlinePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return f"text '{str(wrp_self)}' =>"
    
    
class UniquePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        unique = wrp_params['keys']
        val = f'List of keys: {unique}\n'
        return f"{val}text '{str(wrp_self)}' =>"
    
    
class GenerateTextPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return f"{str(wrp_self)}"
    
    
class ClusterPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        assert isinstance(wrp_self.value, list), "ClusterPreProcessor can only be applied to a list"
        return wrp_self.value


class ForEachPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        val = prep_as_str(wrp_self)
        cond = wrp_params['condition']
        apply = wrp_params['apply']
        return f"{val} foreach '{cond}' apply '{apply}' =>"
    
    
class MapPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        val = prep_as_str(wrp_self)
        context = wrp_params['context']
        return f"{val} map '{str(context)}' =>"
    
    
class ListPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        val = prep_as_str(wrp_self)
        cond = wrp_params['condition']
        return f"{val} list '{cond}' =>"
    

class QueryPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        val = prep_as_str(wrp_self)
        val = f'Context:\n{str(val)}\n'
        query = wrp_params['context']
        return f"{val}{query}\n"

        
class SufficientInformationPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        val = prep_as_str(wrp_self)
        query = prep_as_str(wrp_params['query'])
        return f'query {query} content {val} =>'


class FormatPromptWithArgs0PreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        assert len(args) >= 1
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        wrp_params['prompt'] = wrp_params['prompt'].format(str(args[0]))
        return ''
    
    
class ModifyPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        changes = wrp_params['changes']
        return f"text '{str(wrp_self)}' modify '{str(changes)}'=>"
    
    
class FilterPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        criteria = wrp_params['criteria']
        include = 'include' if wrp_params['include'] else 'remove'
        return f"text '{str(wrp_self)}' {include} '{str(criteria)}' =>"
    

class ArgsToInputPreProcessor(PreProcessor):
    def __init__(self, skip: Optional[List[int]] = None) -> None:
        super().__init__()
        skip = [skip] if skip and isinstance(skip, int) else skip
        self.skip = skip if skip is not None else []
    
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        input_ = ''
        for i, arg in enumerate(args):
            if i in self.skip:
                continue
            input_ += f"{str(arg)}\n"
        return input_
    
    
class SelfToInputPreProcessor(PreProcessor):
    def __init__(self, skip: Optional[List[int]] = None) -> None:
        super().__init__()
        skip = [skip] if skip and isinstance(skip, int) else skip
        self.skip = skip if skip is not None else []
    
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        input_ = f'{str(wrp_self)}\n'
        return input_
    

class DataTemplatePreProcessor(PreProcessor):
    def __init__(self, skip: Optional[List[int]] = None) -> None:
        super().__init__()
        self.skip = skip if skip is not None else []
    
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        input_ = f'[Data]:\n{str(wrp_self)}\n'
        for i, arg in enumerate(args):
            if i in self.skip:
                continue
            input_ += f"{str(arg)}\n"
        return input_


class ConsoleInputPreProcessor(PreProcessor):
    def __init__(self, skip: Optional[List[int]] = None) -> None:
        super().__init__()
        skip = [skip] if skip and isinstance(skip, int) else skip
        self.skip = skip if skip is not None else []
    
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        input_ = f'\n{str(args[0])}\n$> '
        return input_

    
class ConsolePreProcessor(PreProcessor):
    def __init__(self, skip: Optional[List[int]] = None) -> None:
        super().__init__()
        skip = [skip] if skip and isinstance(skip, int) else skip
        self.skip = skip if skip is not None else []
    
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        object_ = f"args: {args}\nkwargs: {kwds}"
        wrp_params['expression'] = wrp_params['expr']
        wrp_params['args'] = args
        wrp_params['kwargs'] = kwds 
        return object_


class LanguagePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        language = wrp_params['language']
        wrp_params['prompt'] = wrp_params['prompt'].format(language)
        return f"{str(wrp_self)}"
    
    
class FormatPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        format_ = wrp_params['format']
        wrp_params['prompt'] = wrp_params['prompt'].format(format_)
        val = prep_as_str(wrp_self)
        return f"text {val} format '{format_}' =>"
    
    
class TranscriptionPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        modify_ = wrp_params['modify']
        val = prep_as_str(wrp_self)
        return f"text {val} modify only '{modify_}' =>"
    
    
class StylePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        description = wrp_params['description']
        text = f'Use the following format: {description}\n'
        libs = ', '.join(wrp_params['libraries'])
        libraries = f"Use the following libraries: {libs}\n"
        content = f'Content:\n{str(wrp_self)}\n'
        return f"{text}{libraries}{content}"
    
    
class UnwrapListSymbolsPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        entries = wrp_params['entries']
        res = []
        # unwrap entries
        for entry in entries:
            if type(entry) is not str:
                res.append(str(entry))
            else:
                res.append(entry)
        wrp_params['entries'] = res
        return ""
    
    
class ExceptionPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        ctxt = prep_as_str(wrp_self)
        ctxt = f"{ctxt}\n" if ctxt and len(ctxt) > 0 else ''
        val = wrp_params['query']
        e = wrp_params['exception']
        exception = "".join(traceback.format_exception_only(type(e), e)).strip()
        return f"context '{val}' exception '{exception}' code'{ctxt}' =>"
    
    
class CorrectionPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        ctxt = prep_as_str(wrp_self)
        ctxt = f"{ctxt}\n" if ctxt and len(ctxt) > 0 else ''
        val = wrp_params['context']
        e = wrp_params['exception']
        exception = "".join(traceback.format_exception_only(type(e), e)).strip()
        return f'context "{val}" exception "{exception}" code "{ctxt}" =>'


class EnumPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return f'[{", ".join([str(x) for x in wrp_params["enum"]])}]\n'


class TextMessagePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return f'Text: {str(wrp_self)}\n'
    
    
class SummaryPreProcessing(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        ctxt = f"Context: {wrp_params['context']} " if 'context' in wrp_params else ''
        return f'{ctxt}Text: {str(wrp_self)}\n'
    
    
class CleanTextMessagePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return f"Text: '{str(wrp_self)}' =>"
    
    
class CrawlPatternPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return ''
    

class PredictionMessagePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return f'Prediction:'
    
    
class ArrowMessagePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return f'{str(wrp_self)} =>'
    
    
class ValuePreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args: Any, **kwds: Any) -> Any:
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return f'{str(wrp_self)}'

