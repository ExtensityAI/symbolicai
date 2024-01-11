import json

from IPython.utils import io

from ...base import Engine
from ...settings import SYMAI_CONFIG
from ....symbol import Result

try:
    from serpapi import GoogleSearch
except:
    GoogleSearch = None


class SearchResult(Result):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(value, **kwargs)
        if 'answer_box' in value.keys() and 'answer' in value['answer_box'].keys():
            self._value = value['answer_box']['answer']
        elif 'answer_box' in value.keys() and 'snippet' in value['answer_box'].keys():
            self._value = value['answer_box']['snippet']
        elif 'answer_box' in value.keys() and 'snippet_highlighted_words' in value['answer_box'].keys():
            self._value = value['answer_box']["snippet_highlighted_words"][0]
        elif 'organic_results' in value and 'snippet' in value["organic_results"][0].keys():
            self._value = value["organic_results"][0]['snippet']
        else:
            self._value = value

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
        json_str = json.dumps(self.raw, indent=2)
        return json_str

    def _repr_html_(self) -> str:
        json_str = json.dumps(self.raw, indent=2)
        return json_str


class SerpApiEngine(Engine):
    def __init__(self):
        super().__init__()
        self.config = SYMAI_CONFIG
        self.api_key = self.config['SEARCH_ENGINE_API_KEY']
        self.engine = self.config['SEARCH_ENGINE_MODEL']

    def id(self) -> str:
        if self.config['SEARCH_ENGINE_API_KEY']:
            if GoogleSearch is None:
                print('SerpApi is not installed. Please install it with `pip install symbolicai[serpapi]`')
            return 'search'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'SEARCH_ENGINE_API_KEY' in kwargs:
            self.api_key = kwargs['SEARCH_ENGINE_API_KEY']
        if 'SEARCH_ENGINE_MODEL' in kwargs:
            self.engine  = kwargs['SEARCH_ENGINE_MODEL']

    def forward(self, argument):
        queries  = argument.prop.prepared_input
        kwargs   = argument.kwargs
        queries_ = queries if isinstance(queries, list) else [queries]
        rsp      = []
        engine   = kwargs['engine'] if 'engine' in kwargs else self.engine

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

        metadata = {}

        output = rsp if isinstance(queries, list) else rsp[0]
        output = [output]
        return output, metadata

    def prepare(self, argument):
        res  = ''
        res += str(argument.prop.query)
        res += str(argument.prop.processed_input)
        argument.prop.prepared_input = res
