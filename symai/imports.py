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
    def __init__(self, module: str, *args, **kwargs):
        super(self).__init__(*args, **kwargs)
        self.module = module
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def exists(module):
        return os.path.exists(f'{BASE_PACKAGE_PATH}/{module}/package.json')

    @staticmethod
    def get_from_github(module):
        # if base package does not exist, create it
        if not os.path.exists(BASE_PACKAGE_PATH):
            os.makedirs(BASE_PACKAGE_PATH)
        # Clone repository
        git_url = f'https://github.com/{module}'
        subprocess.check_call(['git', 'clone', git_url, f'{BASE_PACKAGE_PATH}/{module}'])

        # Install dependencies
        with open(f'{BASE_PACKAGE_PATH}/{module}/package.json') as f:
            pkg = json.load(f)
            for dependency in pkg['dependencies']:
                # Update git_url for the dependency
                git_url_dependency = f'https://github.com/{dependency}'
                if not os.path.exists(f'{BASE_PACKAGE_PATH}/{dependency}'):
                    subprocess.check_call(['git', 'clone', git_url_dependency, f'{BASE_PACKAGE_PATH}/{dependency}'])

        # Install requirements
        if os.path.exists(f'{BASE_PACKAGE_PATH}/{module}/requirements.txt'):
            with open(f'{BASE_PACKAGE_PATH}/{module}/requirements.txt') as f:
                for dependency in f.readlines():
                    subprocess.check_call(['pip', 'install', dependency])

    @staticmethod
    def load_module_class(module):
        with open(f'{BASE_PACKAGE_PATH}/{module}/package.json') as f:
            pkg = json.load(f)
            module_classes = []
            for expr in pkg['expressions']:
                module_path = f'{BASE_PACKAGE_PATH}/{expr["module"].replace("/", ".")}'
                # Determine relative path and adjust namespace
                relative_module_path = module_path.replace(BASE_PACKAGE_PATH, '').replace('/', '.').lstrip('.')
                # Replace with actual username and package_name values
                relative_module_path = module.split('/')[0] + '.' + module.split('/')[1] + '.' + relative_module_path
                module_class = getattr(importlib.import_module(relative_module_path), expr['type'])
                module_classes.append(module_class)
            return module_classes

    def __new__(self, module, auto_clone: bool = True, verbose: bool = False, *args, **kwargs):
        if not Import.exists(module) and auto_clone:
            Import.get_from_github(module)
        with open(f'{BASE_PACKAGE_PATH}/{module}/package.json') as f:
            pkg = json.load(f)
        if 'run' not in pkg:
            raise Exception(f"Module '{module}' has no 'run' expression defined.")
        expr = pkg['run']
        module_path = f'{expr["module"].replace("/", ".")}'
        # Determine relative path and adjust namespace
        relative_module_path = module_path.replace(BASE_PACKAGE_PATH, '').replace('/', '.').lstrip('.')
        # Replace with actual username and package_name values
        relative_module_path = module.split('/')[0] + '.' + module.split('/')[1] + '.' + relative_module_path
        class_ = expr['type']
        if verbose:
            print(f"Loading module '{relative_module_path}.{expr['type']}'")
        module_class = getattr(importlib.import_module(relative_module_path), class_)
        instance = module_class(*args, **kwargs)
        return instance

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def install(module: str):
        if not Import.exists(module):
            Import.get_from_github(module)
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
        if Import.exists(module):
            subprocess.check_call(['git', '-C', f'{BASE_PACKAGE_PATH}/{module.replace(".","/")}', 'pull'])
        else:
            print(f"Module '{module}' not found.")
