from typing import Callable, List

from .base import Engine
from .driver.webclient import connect_browsers, dump_page_source, page_loaded


class CrawlerEngine(Engine):
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

    def forward(self, urls: List[str], patterns: List[str], *args, **kwargs) -> List[str]:
        urls     = urls if isinstance(urls, list) else [urls]
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

        return rsp, metadata

    def prepare(self, args, kwargs, wrp_params):
        if 'url' in wrp_params and 'pattern' in wrp_params:
            wrp_params['urls'] = [str(wrp_params['url'])]
            wrp_params['patterns'] = [wrp_params['pattern']]
        else:
            assert len(kwargs) >= 1 or len(args) >= 1

        # be tolerant to kwarg or arg and assign values of urls and patterns
        # assign urls
        if len(args) >= 1:
            wrp_params['urls'] = args[0]
        elif len(kwargs) >= 1:
            keys = list(kwargs.keys())
            wrp_params['urls'] = kwargs[keys[0]]
        # assign patterns
        if len(args) >= 2:
            wrp_params['patterns'] = args[1]
        elif len(kwargs) >= 2:
            keys = list(kwargs.keys())
            wrp_params['patterns'] = kwargs[keys[1]]
