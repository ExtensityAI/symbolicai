from typing import Callable
from bs4 import BeautifulSoup

from ...base import Engine
from ...driver.webclient import connect_browsers, dump_page_source, page_loaded
from ....symbol import Result


class SeleniumResult(Result):
    def __init__(self, value) -> None:
        super().__init__(value)
        if value is not None:
            self._value = self.extract()

    def extract(self):
        tmp = self.value if isinstance(self.value, list) else [self.value]
        res = []
        for r in tmp:
            if r is None:
                continue
            soup = BeautifulSoup(r, 'html.parser')
            text = soup.getText()
            res.append(text)
        res = None if len(res) == 0 else '\n'.join(res)
        return res


class SeleniumEngine(Engine):
    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.driver_handler = None

    def _init_crawler_engine(self):
        self.driver_handler = connect_browsers(debug=False, proxy=None)

    def get_page_source(self, url: str, pattern: str, script: Callable = None) -> str:
        # deprecated
        driver = self.driver_handler()
        driver.get(url)
        try:
            with page_loaded(driver, pattern, debug=self.debug):
                if script is not None: script(driver)
            return driver.page_source
        except Exception as ex:
            if self.debug: dump_page_source(driver)
            if self.debug: print(ex)
            return None

    def id(self) -> str:
        return 'crawler'

    def forward(self, argument):
        kwargs   = argument.kwargs
        urls     = argument.prop.urls
        patterns = argument.prop.patterns
        urls     = urls if isinstance(urls, list) else [urls]

        # check if all urls start with https:// otherwise add it
        urls = [url if url.startswith('http://') or url.startswith('file://') else 'https://' + url for url in urls]

        patterns = patterns if isinstance(patterns, list) else [patterns]
        assert len(urls) == len(patterns)
        rsp = []

        if self.driver_handler is None:
            self._init_crawler_engine()

        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((urls, patterns))

        for url, p in zip(urls, patterns):
            page = self.get_page_source(url=url, pattern=p)
            rsp.append(page)

        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)

        metadata = {}
        if 'metadata' in kwargs and kwargs['metadata']:
            metadata['kwargs'] = kwargs
            metadata['input']  = (urls, patterns)
            metadata['output'] = rsp
            metadata['urls']   = urls
            metadata['patterns'] = patterns

        rsp = SeleniumResult(rsp)
        return rsp, metadata

    def prepare(self, argument):
        urls     = argument.prop.urls
        patterns = argument.prop.patterns

        # be tolerant to kwarg or arg and assign values of urls and patterns
        # assign urls
        if len(argument.args) >= 1:
            urls = argument.args[0]
        elif len(argument.kwargs) >= 1:
            keys = list(argument.kwargs.keys())
            urls = argument.kwargs[keys[0]]
        # assign patterns
        if len(argument.args) >= 2:
            patterns = argument.args[1]
        elif len(argument.kwargs) >= 2:
            keys = list(argument.kwargs.keys())
            patterns = argument.kwargs[keys[1]]

        argument.prop.processed_input = (urls, patterns)
