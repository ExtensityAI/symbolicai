import importlib
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from .symbol import Expression

__root_dir__  = Path.home() / '.symai/packages/'
BASE_PACKAGE_MODULE = '' # use relative path
BASE_PACKAGE_PATH = str(__root_dir__)
sys.path.append(str(__root_dir__))


class Import(Expression):
    def __init__(self, module: str, auto_clone: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.module = module
        self.module_path = f'{BASE_PACKAGE_MODULE}.{self.module.replace("/",".")}'
        if not self.exists(self.module) and auto_clone:
            self.get_from_github(self.module)
        self._module_class = self.load_module_class()

    def exists(self, module):
        return os.path.exists(f'{BASE_PACKAGE_PATH}/{module}/package.json')

    def get_from_github(self, module):
        # Clone repository
        git_url = f'https://github.com/{module}'
        subprocess.check_call(['git', 'clone', git_url, f'{BASE_PACKAGE_PATH}/{module}'])

        # Install dependencies
        with open(f'{BASE_PACKAGE_PATH}/{module}/package.json') as f:
            pkg = json.load(f)
            for dependency in pkg['dependencies']:
                subprocess.check_call(['git', 'clone', git_url, f'{BASE_PACKAGE_PATH}/{dependency}'])

        # Install requirements
        if os.path.exists(f'{BASE_PACKAGE_PATH}/{module}/requirements.txt'):
            with open(f'{BASE_PACKAGE_PATH}/{module}/requirements.txt') as f:
                for dependency in f.readlines():
                    subprocess.check_call(['pip', 'install', dependency])

    def load_module_class(self):
        with open(f'{BASE_PACKAGE_PATH}/{self.module}/package.json') as f:
            pkg = json.load(f)
            module_classes = []
            for expr in pkg['expressions']:
                module_path = f'{self.module.replace("/", ".")}.{expr["module"].replace("/", ".")}'
                module_class = getattr(importlib.import_module(module_path), expr['type'])
                module_classes.append(module_class)
            return module_classes

    def __call__(self, *args, **kwargs):
        with open(f'{BASE_PACKAGE_PATH}/{self.module}/package.json') as f:
            pkg = json.load(f)
            if 'run' not in pkg:
                raise Exception(f"Module '{self.module}' has no 'run' expression defined.")
            expr = pkg['run']
            module_path = f'{self.module.replace("/", ".")}.{expr["module"].replace("/", ".")}'
            class_ = expr['type']
            module_class = getattr(importlib.import_module(module_path), class_)
            instance = module_class()
            return instance(*args, **kwargs)

    def types(self):
        return self._module_class

    @staticmethod
    def install(module: str):
        if not Import(module, auto_clone=True).exists(module):
            Import(module).get_from_github(module)
            print(f"Module '{module}' installed.")
        else:
            print(f"Module '{module}' already installed.")

    @staticmethod
    def remove(module: str):
        module_path = f'{BASE_PACKAGE_PATH}/{module}'
        if os.path.exists(f'{BASE_PACKAGE_PATH}/{module}'):
            def del_rw(action, name, exc):
                os.chmod(name, stat.S_IWRITE)
                os.remove(name)
            shutil.rmtree(module_path, onerror=del_rw)
            print(f"Removed module '{module}'")
        else:
            print(f"Module '{module}' not found.")
        # check if folder is empty and remove it
        module_path = f'{BASE_PACKAGE_PATH}/{module.split("/")[0]}'
        if os.path.exists(module_path) and not os.listdir(module_path):
            os.rmdir(module_path)
            print(f"Removed empty parent folder '{module_path}'")

    @staticmethod
    def list_installed():
        base_dirs = [dir for dir in os.listdir(BASE_PACKAGE_PATH) if os.path.isdir(f'{BASE_PACKAGE_PATH}/{dir}')]

        sub_dirs = []
        for base_dir in base_dirs:
            full_path = f'{BASE_PACKAGE_PATH}/{base_dir}'
            sub_dirs.extend([f'{base_dir}/{dir}' for dir in os.listdir(full_path) if os.path.isdir(f'{full_path}/{dir}')])

        return sub_dirs

    @staticmethod
    def update(module: str):
        if Import(module, auto_clone=False).exists(module):
            subprocess.check_call(['git', '-C', f'{BASE_PACKAGE_PATH}/{module.replace(".","/")}', 'pull'])
        else:
            print(f"Module '{module}' not found.")
