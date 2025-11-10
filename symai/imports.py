import importlib
import json
import logging
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from loguru import logger

from .backend.settings import HOME_PATH
from .symbol import Expression
from .utils import UserMessage

logging.getLogger("subprocess").setLevel(logging.ERROR)


__root_dir__  = HOME_PATH / 'packages'
BASE_PACKAGE_MODULE = '' # use relative path
BASE_PACKAGE_PATH = __root_dir__
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
        module_path = Path(module)
        if module_path.exists() and module_path.is_dir():
            return (module_path / 'package.json').exists()
        return (BASE_PACKAGE_PATH / module / 'package.json').exists()

    @staticmethod
    def get_from_local(module, local_path):
        """Install a package from a local path.

        Args:
            module: Name of the package in format 'username/repo_name'
            local_path: Path to local package directory
        """
        # If base package does not exist, create it
        BASE_PACKAGE_PATH.mkdir(parents=True, exist_ok=True)

        # Create module directory
        module_path = BASE_PACKAGE_PATH / module
        module_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy files from local path to package directory
        try:
            if module_path.exists():
                shutil.rmtree(module_path)
            shutil.copytree(local_path, module_path)
            logger.info(f"Copied local package from {local_path} to {module_path}")

            # Install dependencies
            package_json = module_path / 'package.json'
            if package_json.exists():
                with package_json.open() as f:
                    pkg = json.load(f)
                    for dependency in pkg.get('dependencies', []):
                        # Update git_url for the dependency
                        git_url_dependency = f'git@github.com:{dependency}.git'
                        dependency_path = BASE_PACKAGE_PATH / dependency
                        if not dependency_path.exists():
                            subprocess.check_call(['git', 'clone', git_url_dependency, str(dependency_path)])

                # Install requirements
                requirements_file = module_path / 'requirements.txt'
                if requirements_file.exists():
                    with requirements_file.open() as f:
                        for dependency in f.readlines():
                            dependency_name = dependency.strip()
                            if dependency_name:
                                subprocess.check_call([sys.executable, '-m', 'pip', 'install', dependency_name])
        except Exception as e:
            logger.error(f"Error installing from local path: {e}")
            raise

    @staticmethod
    def get_from_github(module, submodules=False):
        # if base package does not exist, create it
        BASE_PACKAGE_PATH.mkdir(parents=True, exist_ok=True)

        # Clone repository
        git_url = f'git@github.com:{module}.git'
        clone_cmd = ['git', 'clone']
        if submodules:
            clone_cmd.extend(['--recurse-submodules'])
        clone_cmd.extend([git_url, str(BASE_PACKAGE_PATH / module)])
        subprocess.check_call(clone_cmd)

        # Install dependencies
        package_json = BASE_PACKAGE_PATH / module / 'package.json'
        with package_json.open() as f:
            pkg = json.load(f)
            for dependency in pkg['dependencies']:
                # Update git_url for the dependency
                git_url_dependency = f'git@github.com:{dependency}.git'
                dependency_path = BASE_PACKAGE_PATH / dependency
                if not dependency_path.exists():
                    subprocess.check_call(['git', 'clone', git_url_dependency, str(dependency_path)])

        # Install requirements
        requirements_file = BASE_PACKAGE_PATH / module / 'requirements.txt'
        if requirements_file.exists():
            with requirements_file.open() as f:
                for dependency in f.readlines():
                    subprocess.check_call(['pip', 'install', dependency])

    @staticmethod
    def load_module_class(module):
        module_classes = []
        # Detect if module is a local path
        module_path_obj = Path(module)
        is_local_path = module_path_obj.exists() and module_path_obj.is_dir()

        package_path = module_path_obj if is_local_path else BASE_PACKAGE_PATH / module

        with (package_path / 'package.json').open() as f:
            pkg = json.load(f)
            for expr in pkg['expressions']:
                if is_local_path:
                    # For local modules, we need to add the path to sys.path
                    parent_dir = package_path.parent
                    if str(parent_dir) not in sys.path:
                        sys.path.append(str(parent_dir))
                    # Use local module's name from the directory structure
                    module_name = package_path.name
                    relative_module_path = f"{module_name}.{expr['module'].replace('/', '.')}"
                else:
                    module_parts = module.split('/')
                    relative_module_path = '.'.join([*module_parts, expr['module'].replace('/', '.')])

                try:
                    module_class = getattr(importlib.import_module(relative_module_path), expr['type'])
                    module_classes.append(module_class)
                except (ImportError, ModuleNotFoundError) as e:
                    logger.error(f"Error importing module {relative_module_path}: {e}")
                    raise
        return module_classes

    @staticmethod
    def _normalize_expressions(expressions: list[str] | tuple[str] | str) -> tuple[list[str], bool]:
        if isinstance(expressions, str):
            return [expressions], True
        if isinstance(expressions, (list, tuple)):
            expression_list = list(expressions)
            return expression_list, len(expression_list) == 1
        msg = "Invalid type for 'expressions'. Must be str, list or tuple."
        UserMessage(msg)
        raise Exception(msg)

    @staticmethod
    def load_expression(module, expressions: list[str] | tuple[str] | str) -> list[Expression] | Expression:
        expression_list, return_single = Import._normalize_expressions(expressions)
        expected_count = len(expression_list)
        expression_targets = set(expression_list)
        module_classes = []
        # Detect if module is a local path
        module_path_obj = Path(module)
        is_local_path = module_path_obj.exists() and module_path_obj.is_dir()

        package_path = module_path_obj if is_local_path else BASE_PACKAGE_PATH / module

        if is_local_path:
            parent_dir = package_path.parent
            if str(parent_dir) not in sys.path:
                sys.path.append(str(parent_dir))
            module_name = package_path.name
        else:
            module_parts = module.split('/')

        with (package_path / 'package.json').open() as f:
            pkg = json.load(f)
            for expr in pkg['expressions']:
                relative_module_path = (
                    f"{module_name}.{expr['module'].replace('/', '.')}"
                    if is_local_path
                    else '.'.join([*module_parts, expr['module'].replace('/', '.')])
                )

                if expr['type'] not in expression_targets:
                    continue

                try:
                    module_obj = importlib.import_module(relative_module_path)
                    module_class = getattr(module_obj, expr['type'])
                except (ImportError, ModuleNotFoundError) as e:
                    logger.error(f"Error importing module {relative_module_path}: {e}")
                    raise

                if return_single:
                    return module_class
                module_classes.append(module_class)

        assert len(module_classes) > 0, f"Expression '{expressions}' not found in module '{module}'"
        module_classes_names = [str(class_.__name__) for class_ in module_classes]
        missing_expressions = [expr for expr in expression_list if expr not in module_classes_names]
        assert len(module_classes) == expected_count, f"Not all expressions found in module '{module}'. Could not load {missing_expressions}"
        return module_classes

    def __new__(cls, module, auto_clone: bool = True, verbose: bool = False, local_path: str | None = None,
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
        module_path_obj = Path(module)
        is_local_path = module_path_obj.exists() and module_path_obj.is_dir()

        if is_local_path:
            # If module is a local path
            package_path = module_path_obj
            if not (package_path / 'package.json').exists():
                msg = f"No package.json found in {module}"
                UserMessage(msg)
                raise ValueError(msg)

            with (package_path / 'package.json').open() as f:
                pkg = json.load(f)
        else:
            # Module is a GitHub reference
            if not Import.exists(module) and auto_clone:
                if local_path:
                    Import.get_from_local(module, local_path)
                else:
                    Import.get_from_github(module, submodules)

            with (BASE_PACKAGE_PATH / module / 'package.json').open() as f:
                pkg = json.load(f)
        if 'run' not in pkg:
            msg = f"Module '{module}' has no 'run' expression defined."
            UserMessage(msg)
            raise Exception(msg)
        expr = pkg['run']
        module_rel = expr['module'].replace('/', '.')
        module_parts = module.split('/')
        relative_module_path = '.'.join([*module_parts, module_rel])
        class_ = expr['type']
        if verbose:
            logger.info(f"Loading module '{relative_module_path}.{expr['type']}'")
        module_class = getattr(importlib.import_module(relative_module_path), class_)
        return module_class(*args, **kwargs)

    def __call__(self, *_args, **_kwargs):
        msg = "Cannot call Import class directly. Use Import.load_module_class(module) instead."
        UserMessage(msg)
        raise Exception(msg)

    @staticmethod
    def install(module: str, local_path: str | None = None, submodules: bool = False):
        """Install a package from GitHub or a local path.

        Args:
            module: Name of the package in format 'username/repo_name'
            local_path: Optional path to local package directory
            submodules: Whether to initialize submodules for GitHub repos
        """
        # Determine if module is a local path
        local_path_obj = Path(local_path) if local_path is not None else None
        is_local_path = local_path_obj is not None and local_path_obj.exists() and local_path_obj.is_dir()

        if not Import.exists(module):
            if is_local_path:
                Import.get_from_local(module, str(local_path_obj))
                logger.success(f"Module '{module}' installed from local path.")
            else:
                Import.get_from_github(module, submodules)
                logger.success(f"Module '{module}' installed from GitHub.")
        else:
            logger.info(f"Module '{module}' already installed.")

    @staticmethod
    def remove(module: str):
        # Determine if module is a local path
        module_path_obj = Path(module)
        is_local_path = module_path_obj.exists() and module_path_obj.is_dir()

        if is_local_path:
            # For local path, remove directly
            if module_path_obj.exists():
                def del_rw(_action, name, _exc):
                    path_obj = Path(name)
                    path_obj.chmod(stat.S_IWRITE)
                    path_obj.unlink()
                shutil.rmtree(module_path_obj, onerror=del_rw)
                logger.success(f"Removed local module at '{module}'")
            else:
                logger.error(f"Local module '{module}' not found.")
        else:
            # For GitHub modules, remove from packages directory
            module_path = BASE_PACKAGE_PATH / module
            if module_path.exists():
                def del_rw(_action, name, _exc):
                    path_obj = Path(name)
                    path_obj.chmod(stat.S_IWRITE)
                    path_obj.unlink()
                shutil.rmtree(module_path, onerror=del_rw)
                logger.success(f"Removed module '{module}'")

                # Check if folder is empty and remove it
                parent_path = BASE_PACKAGE_PATH / module.split('/')[0]
                if parent_path.exists() and not any(parent_path.iterdir()):
                    parent_path.rmdir()
                    logger.info(f"Removed empty parent folder '{parent_path}'")
            else:
                logger.error(f"Module '{module}' not found.")

    @staticmethod
    def list_installed():
        if not BASE_PACKAGE_PATH.exists():
            return []
        base_dirs = [entry for entry in BASE_PACKAGE_PATH.iterdir() if entry.is_dir()]

        sub_dirs = []
        for base_dir in base_dirs:
            sub_dirs.extend([f'{base_dir.name}/{entry.name}' for entry in base_dir.iterdir() if entry.is_dir()])

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
            module_path_obj = Path(module)
            is_local_path = module_path_obj.exists() and module_path_obj.is_dir()

            # Use the appropriate path based on whether it's local or not
            module_path = module_path_obj if is_local_path else BASE_PACKAGE_PATH / module.replace(".", "/")

            # Construct the git pull command based on whether submodules should be included
            pull_cmd = ['git', '-C', str(module_path)]
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
