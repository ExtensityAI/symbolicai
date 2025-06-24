import ast
import re
import numpy as np

from collections import namedtuple
from typing import Any
from sklearn.cluster import HDBSCAN


class PostProcessor:
    def __call__(self, response, argument) -> Any:
        raise NotImplementedError()


class StripPostProcessor(PostProcessor):
    def __call__(self, response, argument) -> Any:
        if response is None:
            return None
        if not isinstance(response, str):
            return response
        tmp = response.strip()
        if tmp.startswith("'") and tmp.endswith("'"):
            tmp = tmp[1:-1]
            tmp = tmp.strip()
        return tmp


class ClusterPostProcessor(PostProcessor):
    def __call__(self, response, argument) -> Any:
        assert hasattr(self, 'clustering_kwargs'), "'clustering_kwargs' must be set in the class before calling __call__; use 'set' method."
        clustering = HDBSCAN(**self.clustering_kwargs).fit(response)
        ids = np.unique(clustering.labels_)
        map_ = {}
        for id_ in ids:
            indices = np.where(clustering.labels_ == id_)[0]
            map_[id_] = [argument.prop.instance.value[i] for i in indices]
        return map_

    def set(self, clustering_kwargs):
        self.clustering_kwargs = clustering_kwargs

    def delete(self):
        del self.clustering_kwargs


class TemplatePostProcessor(PostProcessor):
    def __call__(self, response, argument) -> Any:
        template    = argument.prop.template
        placeholder = argument.prop.placeholder
        template    = argument.prop.template
        parts = str(template).split(placeholder)
        return f'{parts[0]}{response}{parts[1]}'


class SplitNewLinePostProcessor(PostProcessor):
    def __call__(self, response, argument) -> Any:
        tmp = response.split('\n')
        return [t.strip() for t in tmp if len(t.strip()) > 0]


class JsonTruncatePostProcessor(PostProcessor):
    def __call__(self, response, argument) -> Any:
        count_b = response.count('[JSON_BEGIN]')
        count_e = response.count('[JSON_END]')
        if count_b > 1 or count_e > 1:
            raise ValueError("More than one [JSON_BEGIN] or [JSON_END] found. Please only generate one JSON response.")
        # cut off everything until the first '{'
        start_idx = response.find('{')
        response = response[start_idx:]
        # find the first occurence of '}' looking backwards
        end_idx = response.rfind('}') + 1
        response = response[:end_idx]
        # search after the first character of '{' if it is a '"' and if not, replace it
        try:
            if response[1:].strip()[0] == "'":
                response = response.replace("'", '"')
        except IndexError:
            pass
        return response


class JsonTruncateMarkdownPostProcessor(PostProcessor):
    def __call__(self, response, argument) -> Any:
        count_b = response.count('```json')
        count_e = response.count('```')
        if count_b > 1 or count_e > 2:
            raise ValueError("More than one ```json Markdown found. Please only generate one JSON response.")
        # cut off everything until the first '{'
        start_idx = response.find('{')
        response = response[start_idx:]
        # find the first occurence of '}' looking backwards
        end_idx = response.rfind('}') + 1
        response = response[:end_idx]
        # search after the first character of '{' if it is a '"' and if not, replace it
        try:
            if response[1:].strip()[0] == "'":
                response = response.replace("'", '"')
        except IndexError:
            pass
        return response


class CodeExtractPostProcessor(PostProcessor):
    def __call__(self, response, argument, tag=None, **kwargs) -> Any:
        if '```' not in str(response):
            return response
        try:
            if tag is None:
                pattern = r'```(?:\w*\n)?(.*?)(?:```|$)'
            else:
                pattern = r'```(?:\w*\n)?' + re.escape(str(tag)) + r'\n(.*?)(?:```|$)'
            matches = re.findall(pattern, str(response), re.DOTALL)
            code = "\n".join(matches).strip()
            return code
        except Exception:
            # If any error occurs during processing, return the original response
            return response


class WolframAlphaPostProcessor(PostProcessor):
    def __call__(self, response, argument) -> Any:
        try:
            res = next(response.value.results).text
            response._value = res
            return response
        except StopIteration:
            vals = ''
            for pod in response.value.pods:
                for sub in pod.subpods:
                    vals += f'{sub.plaintext}\n'
            if len(vals) > 0:
                response._value = vals
                return response
            return response


class SplitPipePostProcessor(PostProcessor):
    def __call__(self, response, argument) -> Any:
        tmp = response if isinstance(response, list) else [response]
        tmp = [r.split('|') for r in tmp if len(r.strip()) > 0]
        tmp = sum(tmp, [])
        return [t.strip() for t in tmp if len(t.strip()) > 0]


class NotifySubscriberPostProcessor(PostProcessor):
    def __call__(self, response, argument) -> Any:
        for k, v in argument.kwargs['subscriber'].items():
            if k in response:
                Event = namedtuple('Event', ['args', 'kwargs', 'response'])
                v(Event(argument.args, argument.kwargs, response))
        return response


class ASTPostProcessor(PostProcessor):
    def __call__(self, response, *args, **kwargs) -> Any:
        try:
            val = ast.literal_eval(response.strip())
            return self._recursive_parse(val)
        except:
            return response

    def _recursive_parse(self, obj):
        """Recursively parse nested structures, converting string representations to actual types."""
        if isinstance(obj, str):
            try:
                return self._recursive_parse(ast.literal_eval(obj))
            except:
                return obj
        elif isinstance(obj, list):
            return [self._recursive_parse(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._recursive_parse(item) for item in obj)
        elif isinstance(obj, set):
            return {self._recursive_parse(item) for item in obj}
        elif isinstance(obj, dict):
            return {key: self._recursive_parse(value) for key, value in obj.items()}
        else:
            return obj


class ConsolePostProcessor(PostProcessor):
    def __call__(self, response, argument) -> Any:
        verbose = argument.prop.verbose
        if verbose: print(f"Argument: {argument}")
        return response


class TakeLastPostProcessor(PostProcessor):
    def __call__(self, response, argument) -> Any:
        return response[-1]


class ExpandFunctionPostProcessor(PostProcessor):
    def __call__(self, response, argument) -> Any:
        return 'def ' + response


class CaseInsensitivePostProcessor(PostProcessor):
    def __call__(self, response, argument) -> Any:
        return str(response).lower()

class ConfirmToBoolPostProcessor(PostProcessor):
    def __call__(self, response, argument) -> Any:
        if response is None:
            return False
        rsp = response.strip()
        from .symbol import Symbol
        sym = Symbol(rsp)
        if sym.isinstanceof('confirming answer'):
            return True
        else:
            return False
