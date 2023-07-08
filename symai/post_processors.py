import ast
import re
from collections import namedtuple
from typing import Any

import numpy as np
from bs4 import BeautifulSoup
from sklearn.cluster import AffinityPropagation


class PostProcessor:
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError()


class StripPostProcessor(PostProcessor):
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
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
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
        clustering = AffinityPropagation().fit(response)
        ids = np.unique(clustering.labels_)
        map_ = {}
        for id_ in ids:
            indices = np.where(clustering.labels_ == id_)[0]
            map_[id_] = [wrp_self.value[i] for i in indices]
        return map_


class TemplatePostProcessor(PostProcessor):
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
        template = wrp_params['template']
        placeholder = wrp_params['placeholder']
        template = wrp_params['template']
        parts = str(template).split(placeholder)
        return f'{parts[0]}{response}{parts[1]}'


class SplitNewLinePostProcessor(PostProcessor):
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
        tmp = response.split('\n')
        return [t.strip() for t in tmp if len(t.strip()) > 0]


class JsonTruncatePostProcessor(PostProcessor):
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
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


class CodeExtractPostProcessor(PostProcessor):
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
        if '```' not in response:
            return response
        matches = []
        try:
            pattern = r'```(?:\w*\n)?(.*?)```'
            matches = re.findall(pattern, response, re.DOTALL)
        except IndexError:
            pass
        code = "\n".join(matches).strip()
        return code


class WolframAlphaPostProcessor(PostProcessor):
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
        try:
            return next(response.results).text
        except StopIteration:
            vals = ''
            for pod in response.pods:
                for sub in pod.subpods:
                    vals += f'{sub.plaintext}\n'
            if len(vals) > 0:
                return vals
            return response


class SplitPipePostProcessor(PostProcessor):
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
        tmp = response if isinstance(response, list) else [response]
        tmp = [r.split('|') for r in tmp if len(r.strip()) > 0]
        tmp = sum(tmp, [])
        return [t.strip() for t in tmp if len(t.strip()) > 0]


class NotifySubscriberPostProcessor(PostProcessor):
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
        for k, v in wrp_params['subscriber'].items():
            if k in response:
                Event = namedtuple('Event', ['args', 'kwargs', 'response'])
                v(Event(args, kwds, response))
        return response


class ASTPostProcessor(PostProcessor):
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
        try:
            val = ast.literal_eval(response.strip())
            return val
        except:
            return response


class ConsolePostProcessor(PostProcessor):
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
        verbose = response['verbose'] if 'verbose' in response else False
        input_ = response['input'] if 'input' in response else None
        if verbose: print(f"Input: {input_}")
        expr_ = response['expr'] if 'expr' in response else None
        if verbose: print(f"Expression: {expr_}")
        args_kwargs = (response['args'], response['kwargs'])
        if verbose: print(f"args: {args_kwargs[0]} kwargs: {args_kwargs[1]}")
        rsp = f"Dictionary: {response}"
        if verbose: print(rsp)
        output = f"Output: {response['output']}"
        if verbose: print(output)
        return response['output']


class TakeLastPostProcessor(PostProcessor):
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
        return response[-1]


class ExpandFunctionPostProcessor(PostProcessor):
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
        return 'def ' + response


class CaseInsensitivePostProcessor(PostProcessor):
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
        return response.lower()


class HtmlGetTextPostProcessor(PostProcessor):
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> Any:
        tmp = response if isinstance(response, list) else [response]
        res = []
        for r in tmp:
            if r is None:
                continue
            soup = BeautifulSoup(r, 'html.parser')
            text = soup.getText()
            res.append(text)
        res = None if len(res) == 0 else '\n'.join(res)
        return res

class ConfirmToBoolPostProcessor(PostProcessor):
    def __call__(self, wrp_self, wrp_params, response, *args: Any, **kwds: Any) -> bool:
        if response is None:
            return False
        rsp = response.strip()
        from .symbol import Symbol
        sym = Symbol(rsp)
        if sym.isinstanceof('confirming answer'):
            return True
        else:
            return False
