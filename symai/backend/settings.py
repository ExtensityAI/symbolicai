import json
import os
import sys
from pathlib import Path


class SymAIConfig:
    """Manages SymbolicAI configuration files across different environments.
    Configuration Priority:
    1. Debug mode (current working directory)
    2. Python environment-specific config
    3. User home directory config
    """

    def __init__(self):
        """Initialize configuration paths based on current Python environment."""
        self._env_path = Path(sys.prefix)
        self._env_config_dir = self._env_path / '.symai'
        self._home_config_dir = Path.home() / '.symai'
        self._debug_dir = Path.cwd()  # Current working directory for debug mode

    @property
    def config_dir(self) -> Path:
        """Returns the active configuration directory based on priority system."""
        # Debug mode takes precedence
        if (self._debug_dir / 'symai.config.json').exists():
            return self._debug_dir
        # Then environment config
        if self._env_config_dir.exists():
            return self._env_config_dir
        # Finally home directory
        return self._home_config_dir

    def get_config_path(self, filename: str, fallback_to_home: bool = False) -> Path:
        """Gets the config path using the priority system or forces fallback to home."""
        debug_config = self._debug_dir / filename
        env_config   = self._env_config_dir / filename
        home_config  = self._home_config_dir / filename

        # Check debug first (only valid for symai.config.json)
        if filename == 'symai.config.json' and debug_config.exists():
            return debug_config

        # If forced to fallback, return home config if it exists, otherwise environment
        if fallback_to_home:
            return home_config if home_config.exists() else env_config

        # Normal priority-based resolution
        # If environment config doesn't exist, return that path (for creation)
        if not env_config.exists():
            return env_config
        # Otherwise use environment config
        return env_config

    def load_config(self, filename: str, fallback_to_home: bool = False) -> dict:
        """Loads JSON data from the determined config location."""
        config_path = self.get_config_path(filename, fallback_to_home=fallback_to_home)
        if not config_path.exists():
            return {}
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_config(self, filename: str, data: dict, fallback_to_home: bool = False) -> None:
        """Saves JSON data to the determined config location."""
        config_path = self.get_config_path(filename, fallback_to_home=fallback_to_home)
        os.makedirs(config_path.parent, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def migrate_config(self, filename: str, updates: dict) -> None:
        """Updates existing configuration with new fields."""
        config = self.load_config(filename)
        config.update(updates)
        self.save_config(filename, config)

SYMAI_CONFIG = {}
SYMSH_CONFIG = {}
SYMSERVER_CONFIG = {}
HOME_PATH = Path(SymAIConfig().config_dir)
