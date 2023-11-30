import os

os.environ['WDM_LOG'] = 'false' # disable webdriver-manager logging

import logging

logging.getLogger('WDM').setLevel(logging.NOTSET)

import random
import re
import sys
import time
import urllib.request
import warnings

from random import choice

try:
    warnings.filterwarnings('ignore', module='chromedriver')
    warnings.filterwarnings('ignore', module='selenium')
    from selenium import webdriver
    from selenium.webdriver.remote.remote_connection import LOGGER
    from webdriver_manager.firefox import GeckoDriverManager
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.firefox.service import Service as FirefoxService
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from webdriver_manager.microsoft import EdgeChromiumDriverManager
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.edge.service import Service as EdgeService
    from selenium.common.exceptions import WebDriverException

    LOGGER.setLevel(logging.ERROR)

except Exception as e:
    webdriver = None
    if "No module named 'selenium'" in str(e):
        print(f"ERROR: {e}")
        print(f"ERROR: Please install selenium with `pip install selenium`")
    else:
        print(f"ERROR: {e}")

from ... import __root_dir__


class Proxy(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port


class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


def wait_for(condition_function, timeout):
    start_time = time.time()
    while time.time() < start_time + timeout:
        if condition_function():
            return True
        else:
            time.sleep(0.1)
    raise Exception(f'Server does not respond to request with appropriate content. Check link or script.')


class page_loaded(object):
    def __init__(self, driver, check_pattern, timeout=3, debug=False):
        self.check_pattern = check_pattern
        self.driver = driver
        self.timeout = timeout
        self.debug = debug

    def __enter__(self):
        pass

    def page_has_loaded(self):
        if self.debug: print(self.driver.page_source)
        return re.search(self.check_pattern, self.driver.page_source)

    def __exit__(self, *_):
        wait_for(self.page_has_loaded, timeout=self.timeout)


def add_options(options, proxy):
    if proxy: options.add_argument(f"--proxy-server=socks5://{proxy.host}:{proxy.port}")
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')
    options.add_argument("--headless")
    options.add_argument("--log-level=3")


def _connect_brower(debug, proxy=None):
    assert webdriver is not None, "selenium is not installed"

    try:
        options = ChromeOptions()
        add_options(options, proxy)
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    except Exception as e1:
        try:
            print(f"ERROR REMEDY: Trying to use Chrome as an alternative.")
            options = FirefoxOptions()
            add_options(options, proxy)
            driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)
        except Exception as e2:
            print(f"Issue with finding an appropriate driver version. Your current browser might be newer than the driver. Please either downgrade Chrome or try to install a proper chromedriver manually.\nOriginal error: {e1}; Remedy attempt error: {e2}")
            try:
                print(f"ERROR REMEDY: Trying to use Edge as an alternative.")
                options = EdgeOptions()
                if proxy: options.add_argument(f"--proxy-server=socks5://{proxy.host}:{proxy.port}")
                add_options(options, proxy)
                driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()), options=options)
            except Exception as e3:
                print(f"Issue with finding an appropriate driver version. Your current browser might be newer than the driver. Please either downgrade Chrome or try to install a proper chromedriver manually.\nOriginal error: {e1}; Remedy attempt error: {e2}; Second remedy attempt error: {e3}")
                raise e3 from e2
    if debug: print("Chrome Headless Browser Invoked")

    return driver


def connect_browsers(debug, proxy):
    assert webdriver is not None, "selenium is not installed"
    class BrowserHandler(object):
        def __init__(self, debug):
            self.browsers = [_connect_brower(debug, proxy=proxy)]
        def __call__(self):
            return choice(self.browsers)
    return BrowserHandler(debug)


def dump_page_source(driver, file_path="dump.log"):
    with open(file_path, "wb") as err_file:
        err_file.write(driver.page_source.encode("utf-8"))



def contains_text(check_pattern, search_pattern, link, driver_handler, script=None, debug=False, args=None):
    driver = driver_handler()
    try:
        driver.get(link)
    except WebDriverException as e:
        print(f"ERROR: {e}")
        print(f"ERROR: Please install a proper driver for your browser. You can find the appropriate driver here: https://selenium-python.readthedocs.io/installation.html#drivers")
        return False
    with page_loaded(driver, check_pattern, debug=debug):
        if script is not None: script(driver, args)
    rsp = re.search(search_pattern, driver.page_source)
    return True if rsp else False


def download_images(driver, args):
    src = driver.find_element_by_xpath("//div[@class='bRMDJf islir']/descendant-or-self::*[img]").get_attribute("src")
    imgs = driver.find_elements_by_tag_name('img')
    p = args[0]
    uri = []
    for i in imgs:
        data = i.get_attribute("src")
        if data and data.startswith('data'):
            response = urllib.request.urlopen(data)
            n = i.get_attribute("data-iml")
            if n:
                if not os.path.exists(f'data/{p}'):
                    os.makedirs(f'data/{p}')
                with open(f'data/{p}/{n}.jpg', 'wb') as f:
                    f.write(response.file.read())


def run_selenium_test(debug=False):
    proxy = Proxy(host="localhost", port=9050)
    driver_handler = connect_browsers(debug, proxy)

    max_cnt = 5
    cnt = 0
    now = time.time()

    with open('data/text_demo.txt', 'r') as f:
        text = f.read().replace('\n', '')\
                       .replace('.', '')\
                       .replace(',', '')

    print(text)
    #text = "women sitting in the park" #names.get_full_name()

    # input String
    lst = text.split(' ')

    while cnt < max_cnt:
        start_pos = random.randint(0, len(lst)-1)
        len_ = random.randint(1, 6)
        seq = lst[start_pos:start_pos+len_]
        pattern = " ".join(seq)
        result = contains_text(check_pattern="Images",
                               search_pattern=pattern,
                               link=f"https://www.google.com/search?tbm=isch&as_q={pattern}&tbs=isz:lt,islt:4mp,sur:fmc",
                               driver_handler=driver_handler,
                               debug=True,
                               script=download_images,
                               args=[pattern])
        timestamp = time.time() - now
        now = time.time()
        print(f"{result} - time: {timestamp}")
        cnt += 1


if __name__ == "__main__":
    run_selenium_test(debug=True)
