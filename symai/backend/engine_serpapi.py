from typing import List
from IPython.utils import io
from box import Box
import json

from .base import Engine
from .settings import SYMAI_CONFIG
from .. import Symbol

try:
    from serpapi import GoogleSearch
except:
    GoogleSearch = None
    print('SerpApi is not installed. Please install it with `pip install symbolicai[serpapi]`')



class SearchResult(Symbol):
    def __init__(self, value) -> None:
        super().__init__(value)
        self.raw = Box(value)
        if 'answer_box' in value.keys() and 'answer' in value['answer_box'].keys():
            self.value = value['answer_box']['answer']
        elif 'answer_box' in value.keys() and 'snippet' in value['answer_box'].keys():
            self.value = value['answer_box']['snippet']
        elif 'answer_box' in value.keys() and 'snippet_highlighted_words' in value['answer_box'].keys():
            self.value = value['answer_box']["snippet_highlighted_words"][0]
        elif 'organic_results' in value and 'snippet' in value["organic_results"][0].keys():
            self.value= value["organic_results"][0]['snippet']
        else:
            self.value = value

        if 'organic_results' in value.keys():
            self.results = value['organic_results']
            if len(self.results) > 0:
                self.links = [r['link'] for r in self.results]
            else:
                self.links = []
        else:
            self.results = []
            self.links = []

    def __str__(self) -> str:
        json_str = json.dumps(self.raw.to_dict(), indent=2)
        return json_str

    def _repr_html_(self) -> str:
        json_str = json.dumps(self.raw.to_dict(), indent=2)
        return json_str


class SerpApiEngine(Engine):
    def __init__(self):
        super().__init__()
        config = SYMAI_CONFIG
        self.api_key = config['SEARCH_ENGINE_API_KEY']
        self.engine = config['SEARCH_ENGINE_MODEL']

    def command(self, wrp_params):
        super().command(wrp_params)
        if 'SEARCH_ENGINE_API_KEY' in wrp_params:
            self.api_key = wrp_params['SEARCH_ENGINE_API_KEY']
        if 'SEARCH_ENGINE_MODEL' in wrp_params:
            self.engine  = wrp_params['SEARCH_ENGINE_MODEL']

    def forward(self, queries: List[str], *args, **kwargs) -> List[str]:
        queries_ = queries if isinstance(queries, list) else [queries]
        rsp      = []
        engine   = kwargs['engine'] if 'engine' in kwargs else self.engine

        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((queries_,))

        for q in queries_:
            query = {
                "api_key": self.api_key,
                "engine": engine,
                "q": q,
                "google_domain": "google.com",
                "gl": "us",
                "hl": "en"
            }

            # send to Google
            with io.capture_output() as captured: # disables prints from GoogleSearch
                search = GoogleSearch(query)
                res = search.get_dict()

            toret = SearchResult(res)
            rsp.append(toret)

        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = queries_
            metadata['output'] = rsp
            metadata['model']  = self.engine

        output = rsp if isinstance(queries, list) else rsp[0]
        return output, metadata

    def prepare(self, args, kwargs, wrp_params):
        wrp_params['queries'] = [str(wrp_params['query'])]
