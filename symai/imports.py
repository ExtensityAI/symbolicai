import importlib
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Union

from loguru import logger

from .backend.settings import HOME_PATH
from .symbol import Expression

logging.getLogger("subprocess").setLevel(logging.ERROR)


__root_dir__  = HOME_PATH / 'packages'
BASE_PACKAGE_MODULE = '' # use relative path
BASE_PACKAGE_PATH = str(__root_dir__)
sys.path.append(str(__root_dir__))


class Import(Expression):
    def __init__(self, module: str, local_path: str | None = None, submodules: bool = False, *args, **kwargs):
        super(self).__init__(*args, **kwargs)
        self.module = module
        self.local_path = local_path
        self.submodules = submodules

    @staticmethod
    def exists(module):
        # Check if module is a local path or a GitHub repo reference
        if os.path.exists(module) and os.path.isdir(module):
            return os.path.exists(f'{module}/package.json')
        else:
            return os.path.exists(f'{BASE_PACKAGE_PATH}/{module}/package.json')

    @staticmethod
    def get_from_local(module, local_path):
        """Install a package from a local path.

        Args:
            module: Name of the package in format 'username/repo_name'
            local_path: Path to local package directory
        """
        # If base package does not exist, create it
        if not os.path.exists(BASE_PACKAGE_PATH):
            os.makedirs(BASE_PACKAGE_PATH)

        # Create module directory
        module_path = f'{BASE_PACKAGE_PATH}/{module}'
        os.makedirs(os.path.dirname(module_path), exist_ok=True)

        # Copy files from local path to package directory
        try:
            if os.path.exists(module_path):
                shutil.rmtree(module_path)
            shutil.copytree(local_path, module_path)
            logger.info(f"Copied local package from {local_path} to {module_path}")

            # Install dependencies
            if os.path.exists(f'{module_path}/package.json'):
                with open(f'{module_path}/package.json') as f:
                    pkg = json.load(f)
                    for dependency in pkg.get('dependencies', []):
                        # Update git_url for the dependency
                        git_url_dependency = f'git@github.com:{dependency}.git'
                        if not os.path.exists(f'{BASE_PACKAGE_PATH}/{dependency}'):
                            subprocess.check_call(['git', 'clone', git_url_dependency, f'{BASE_PACKAGE_PATH}/{dependency}'])

                # Install requirements
                if os.path.exists(f'{module_path}/requirements.txt'):
                    with open(f'{module_path}/requirements.txt') as f:
                        for dependency in f.readlines():
                            dependency = dependency.strip()
                            if dependency:
                                subprocess.check_call([sys.executable, '-m', 'pip', 'install', dependency])
        except Exception as e:
            logger.error(f"Error installing from local path: {e}")
            raise

    @staticmethod
    def get_from_github(module, submodules=False):
        # if base package does not exist, create it
        if not os.path.exists(BASE_PACKAGE_PATH):
            os.makedirs(BASE_PACKAGE_PATH)

        # Clone repository
        git_url = f'git@github.com:{module}.git'
        clone_cmd = ['git', 'clone']
        if submodules:
            clone_cmd.extend(['--recurse-submodules'])
        clone_cmd.extend([git_url, f'{BASE_PACKAGE_PATH}/{module}'])
        subprocess.check_call(clone_cmd)

        # Install dependencies
        with open(f'{BASE_PACKAGE_PATH}/{module}/package.json') as f:
            pkg = json.load(f)
            for dependency in pkg['dependencies']:
                # Update git_url for the dependency
                git_url_dependency = f'git@github.com:{dependency}.git'
                if not os.path.exists(f'{BASE_PACKAGE_PATH}/{dependency}'):
                    subprocess.check_call(['git', 'clone', git_url_dependency, f'{BASE_PACKAGE_PATH}/{dependency}'])

        # Install requirements
        if os.path.exists(f'{BASE_PACKAGE_PATH}/{module}/requirements.txt'):
            with open(f'{BASE_PACKAGE_PATH}/{module}/requirements.txt') as f:
                for dependency in f.readlines():
                    subprocess.check_call(['pip', 'install', dependency])

    @staticmethod
    def load_module_class(module):
        module_classes = []
        # Detect if module is a local path
        is_local_path = os.path.exists(module) and os.path.isdir(module)

        if is_local_path:
            package_path = module
        else:
            package_path = f'{BASE_PACKAGE_PATH}/{module}'

        with open(f'{package_path}/package.json') as f:
            pkg = json.load(f)
            for expr in pkg['expressions']:
                if is_local_path:
                    module_path = f'{package_path}/{expr["module"].replace("/", ".")}'
                    # For local modules, we need to add the path to sys.path
                    parent_dir = os.path.dirname(package_path)
                    if parent_dir not in sys.path:
                        sys.path.append(parent_dir)
                    # Use local module's name from the directory structure
                    module_name = os.path.basename(package_path)
                    relative_module_path = f"{module_name}.{expr['module'].replace('/', '.')}"
                else:
                    module_path = f'{BASE_PACKAGE_PATH}/{expr["module"].replace("/", ".")}'
                    # Determine relative path and adjust namespace
                    relative_module_path = module_path.replace(BASE_PACKAGE_PATH, '').replace('/', '.').lstrip('.')
                    # Replace with actual username and package_name values
                    relative_module_path = module.split('/')[0] + '.' + module.split('/')[1] + '.' + relative_module_path

                try:
                    module_class = getattr(importlib.import_module(relative_module_path), expr['type'])
                    module_classes.append(module_class)
                except (ImportError, ModuleNotFoundError) as e:
                    logger.error(f"Error importing module {relative_module_path}: {e}")
                    raise
        return module_classes

    @staticmethod
    def load_expression(module, expressions: Union[List[str] | Tuple[str] | str]) -> Union[List[Expression] | Expression]:
        module_classes = []
        # Detect if module is a local path
        is_local_path = os.path.exists(module) and os.path.isdir(module)

        if is_local_path:
            package_path = module
        else:
            package_path = f'{BASE_PACKAGE_PATH}/{module}'

        with open(f'{package_path}/package.json') as f:
            pkg = json.load(f)
            for expr in pkg['expressions']:
                if is_local_path:
                    # For local paths, need to handle imports differently
                    parent_dir = os.path.dirname(package_path)
                    if parent_dir not in sys.path:
                        sys.path.append(parent_dir)

                    # Use basename as module name
                    module_name = os.path.basename(package_path)
                    relative_module_path = f"{module_name}.{expr['module'].replace('/', '.')}"
                else:
                    # For GitHub packages
                    module_path = f'{BASE_PACKAGE_MODULE}/{expr["module"].replace("/", ".")}'
                    # Determine relative path and adjust namespace
                    relative_module_path = module_path.replace(BASE_PACKAGE_MODULE, '').replace('/', '.').lstrip('.')
                    # Add the organization/repo prefix
                    relative_module_path = f"{module.split('/')[0]}.{module.split('/')[1]}.{relative_module_path}"

                if isinstance(expressions, str):
                    if expr['type'] == expressions:
                        try:
                            module_obj = importlib.import_module(relative_module_path)
                            module_class = getattr(module_obj, expr['type'])
                            return module_class
                        except (ImportError, ModuleNotFoundError) as e:
                            logger.error(f"Error importing module {relative_module_path}: {e}")
                            raise
                elif isinstance(expressions, list) or isinstance(expressions, tuple):
                    if expr['type'] in expressions:
                        try:
                            module_obj = importlib.import_module(relative_module_path)
                            module_class = getattr(module_obj, expr['type'])
                            if len(expressions) == 1:
                                return module_class
                            module_classes.append(module_class)
                        except (ImportError, ModuleNotFoundError) as e:
                            logger.error(f"Error importing module {relative_module_path}: {e}")
                            raise
                else:
                    raise Exception("Invalid type for 'expressions'. Must be str, list or tuple.")

        assert len(module_classes) > 0, f"Expression '{expressions}' not found in module '{module}'"
        module_classes_names = [str(class_.__name__) for class_ in module_classes]
        assert len(module_classes) == len(expressions), f"Not all expressions found in module '{module}'. Could not load {[expr for expr in expressions if expr not in module_classes_names]}"
        return module_classes

    def __new__(self, module, auto_clone: bool = True, verbose: bool = False, local_path: str | None = None,
                submodules: bool = False, *args, **kwargs):
        """
        Import a module from GitHub or local path.

        Args:
            module: Either a GitHub reference (username/repo) or a local path
            auto_clone: Whether to automatically clone from GitHub if needed
            verbose: Whether to logger.info verbose information
            local_path: Path to local package directory
            submodules: Whether to initialize submodules for GitHub repos
            *args, **kwargs: Additional arguments to pass to the module constructor
        """
        # Detect if module is a local path
        is_local_path = os.path.exists(module) and os.path.isdir(module)

        if is_local_path:
            # If module is a local path
            package_path = module
            if not os.path.exists(f'{package_path}/package.json'):
                raise ValueError(f"No package.json found in {module}")

            with open(f'{package_path}/package.json') as f:
                pkg = json.load(f)
        else:
            # Module is a GitHub reference
            if not Import.exists(module) and auto_clone:
                if local_path:
                    Import.get_from_local(module, local_path)
                else:
                    Import.get_from_github(module, submodules)

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
            logger.info(f"Loading module '{relative_module_path}.{expr['type']}'")
        module_class = getattr(importlib.import_module(relative_module_path), class_)
        return module_class(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        raise Exception("Cannot call Import class directly. Use Import.load_module_class(module) instead.")

    @staticmethod
    def install(module: str, local_path: str = None, submodules: bool = False):
        """Install a package from GitHub or a local path.

        Args:
            module: Name of the package in format 'username/repo_name'
            local_path: Optional path to local package directory
            submodules: Whether to initialize submodules for GitHub repos
        """
        # Determine if module is a local path
        is_local_path = local_path is not None and os.path.exists(local_path) and os.path.isdir(local_path)

        if not Import.exists(module):
            if is_local_path:
                Import.get_from_local(module, local_path)
                logger.success(f"Module '{module}' installed from local path.")
            else:
                Import.get_from_github(module, submodules)
                logger.success(f"Module '{module}' installed from GitHub.")
        else:
            logger.info(f"Module '{module}' already installed.")

    @staticmethod
    def remove(module: str):
        # Determine if module is a local path
        is_local_path = os.path.exists(module) and os.path.isdir(module)

        if is_local_path:
            # For local path, remove directly
            if os.path.exists(module):
                def del_rw(action, name, exc):
                    os.chmod(name, stat.S_IWRITE)
                    os.remove(name)
                shutil.rmtree(module, onerror=del_rw)
                logger.success(f"Removed local module at '{module}'")
            else:
                logger.error(f"Local module '{module}' not found.")
        else:
            # For GitHub modules, remove from packages directory
            module_path = f'{BASE_PACKAGE_PATH}/{module}'
            if os.path.exists(module_path):
                def del_rw(action, name, exc):
                    os.chmod(name, stat.S_IWRITE)
                    os.remove(name)
                shutil.rmtree(module_path, onerror=del_rw)
                logger.success(f"Removed module '{module}'")

                # Check if folder is empty and remove it
                parent_path = f'{BASE_PACKAGE_PATH}/{module.split("/")[0]}'
                if os.path.exists(parent_path) and not os.listdir(parent_path):
                    os.rmdir(parent_path)
                    logger.info(f"Removed empty parent folder '{parent_path}'")
            else:
                logger.error(f"Module '{module}' not found.")

    @staticmethod
    def list_installed():
        base_dirs = [dir for dir in os.listdir(BASE_PACKAGE_PATH) if os.path.isdir(f'{BASE_PACKAGE_PATH}/{dir}')]

        sub_dirs = []
        for base_dir in base_dirs:
            full_path = f'{BASE_PACKAGE_PATH}/{base_dir}'
            sub_dirs.extend([f'{base_dir}/{dir}' for dir in os.listdir(full_path) if os.path.isdir(f'{full_path}/{dir}')])

        return sub_dirs

    @staticmethod
    def update(module: str, submodules: bool = False):
        """Update a package from GitHub.

        Args:
            module: Name of the package in format 'username/repo_name' or a local path
            submodules: Whether to update submodules as well
        """
        if Import.exists(module):
            # Determine if module is a local path
            is_local_path = os.path.exists(module) and os.path.isdir(module)

            # Use the appropriate path based on whether it's local or not
            module_path = module if is_local_path else f'{BASE_PACKAGE_PATH}/{module.replace(".","/")}'

            # Construct the git pull command based on whether submodules should be included
            pull_cmd = ['git', '-C', module_path]
            if submodules:
                pull_cmd.extend(['pull', '--recurse-submodules'])
                subprocess.check_call(pull_cmd)
                logger.success(f"Module '{module}' and its submodules updated.")
            else:
                pull_cmd.extend(['pull'])
                subprocess.check_call(pull_cmd)
                logger.success(f"Module '{module}' updated.")
        else:
            logger.warning(f"Module '{module}' not found.")
