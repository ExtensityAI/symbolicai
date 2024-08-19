import os
from pathlib import Path


SYMAI_CONFIG = {}
SYMSH_CONFIG = {}
SYMSERVER_CONFIG = {}
HOME_PATH = Path(os.environ.get('SYMAI_HOME')) if os.environ.get('SYMAI_HOME') else Path.home()
