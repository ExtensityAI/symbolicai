from typing import Callable
from bs4 import BeautifulSoup

from ...base import Engine
from ...driver.webclient import connect_browsers, dump_page_source, page_loaded
from ....symbol import Result


class SeleniumResult(Result):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(value, **kwargs)
        if value is not None:
            self.raw    = value
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
        try:
            driver.get(url)
            with page_loaded(driver, pattern, debug=self.debug):
                if script: driver.execute_script(script)
            return driver.page_source
        except Exception as ex:
            if self.debug: dump_page_source(driver)
            if self.debug: print(ex)
            return f"Sorry, I cannot find the page you are looking for: {ex}"

    def id(self) -> str:
        return 'crawler'

    def forward(self, argument):
        urls, patterns  = argument.prop.prepared_input
        urls     = urls if isinstance(urls, list) else [urls]

        patterns = patterns if isinstance(patterns, list) else [patterns]
        assert len(urls) == len(patterns)
        rsp = []

        self._init_crawler_engine()

        for url, p in zip(urls, patterns):
            page = self.get_page_source(url=url, pattern=p)
            rsp.append(page)

        metadata = {}
        rsp = SeleniumResult(rsp)
        return [rsp], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "CrawlerEngine does not support processed_input."
        assert argument.prop.urls, "CrawlerEngine requires urls."

        argument.prop.urls      = [str(argument.prop.url)]
        argument.prop.patterns  = [str(argument.prop.pattern)]

        # be tolerant to kwarg or arg and assign values of urls and patterns
        # assign urls
        if len(argument.args) >= 1:
            argument.prop.urls = argument.args[0]
        elif len(argument.kwargs) >= 1:
            keys = list(argument.kwargs.keys())
            argument.prop.urls = argument.kwargs[keys[0]]
        # assign patterns
        if len(argument.args) >= 2:
            argument.prop.patterns = argument.args[1]
        elif len(argument.kwargs) >= 2:
            keys = list(argument.kwargs.keys())
            argument.prop.patterns = argument.kwargs[keys[1]]

        argument.prop.prepared_input = (argument.prop.urls, argument.prop.patterns)
