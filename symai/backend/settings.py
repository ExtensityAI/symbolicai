import json
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
        self._active_paths: dict[str, Path] = {}

    def _canonical_key(self, filename: str | Path) -> str:
        """Return a canonical identifier for config files regardless of input type."""
        path = Path(filename)
        if path.is_absolute() or path.parent != Path():
            return str(path)
        return path.name or str(path)

    def _remove_legacy_path_keys(self, key: str) -> None:
        """Drop legacy Path keys that collide with the canonical key."""
        target_path = Path(key)
        target_name = target_path.name or key
        stale_keys: list[Path] = [
            existing_key for existing_key in self._active_paths
            if isinstance(existing_key, Path)
            and (existing_key.name == target_name or str(existing_key) == key)
        ]
        for stale_key in stale_keys:
            self._active_paths.pop(stale_key, None)

    @property
    def config_dir(self) -> Path:
        """Returns the active configuration directory based on priority system."""
        # Debug mode takes precedence
        if (self._debug_dir / 'symai.config.json').exists():
            return self._debug_dir / '.symai'
        # Then environment config
        if self._env_config_dir.exists():
            return self._env_config_dir
        # Finally home directory
        return self._home_config_dir

    def get_config_path(self, filename: str | Path, fallback_to_home: bool = False) -> Path:
        """Gets the config path using the priority system or forces fallback to home."""
        input_path = Path(filename)
        if input_path.is_absolute() or input_path.parent != Path():
            return input_path

        normalized_filename = self._canonical_key(filename)
        # Only use the basename for managed directories
        normalized_filename = Path(normalized_filename).name
        debug_config = self._debug_dir / normalized_filename
        env_config   = self._env_config_dir / normalized_filename
        home_config  = self._home_config_dir / normalized_filename

        # Check debug first (only valid for symai.config.json)
        if normalized_filename == 'symai.config.json' and debug_config.exists():
            return debug_config

        # If forced to fallback, return home config if it exists, otherwise environment
        if fallback_to_home:
            if home_config.exists():
                return home_config
            return env_config

        # Normal priority-based resolution
        if env_config.exists():
            return env_config
        if home_config.exists():
            return home_config
        return env_config

    def load_config(self, filename: str | Path, fallback_to_home: bool = False) -> dict:
        """Loads JSON data from the determined config location."""
        config_path = self.get_config_path(filename, fallback_to_home=fallback_to_home)
        key = self._canonical_key(filename)
        if not config_path.exists():
            self._remove_legacy_path_keys(key)
            self._active_paths.pop(key, None)
            return {}
        with config_path.open(encoding='utf-8') as f:
            config = json.load(f)
        self._remove_legacy_path_keys(key)
        self._active_paths[key] = config_path
        return config

    def save_config(self, filename: str | Path, data: dict, fallback_to_home: bool = False) -> None:
        """Saves JSON data to the determined config location."""
        config_path = self.get_config_path(filename, fallback_to_home=fallback_to_home)
        key = self._canonical_key(filename)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        self._remove_legacy_path_keys(key)
        self._active_paths[key] = config_path

    def migrate_config(self, filename: str, updates: dict) -> None:
        """Updates existing configuration with new fields."""
        config = self.load_config(filename)
        config.update(updates)
        self.save_config(filename, config)

    def get_active_path(self, filename: str | Path) -> Path:
        """Returns the last path used to read or write the given config file."""
        key = self._canonical_key(filename)
        cached = self._active_paths.get(key)
        if cached is not None:
            return cached
        for legacy_key, cached_path in list(self._active_paths.items()):
            if isinstance(legacy_key, Path) and (
                legacy_key.name == key or str(legacy_key) == key
            ):
                self._active_paths.pop(legacy_key, None)
                self._active_paths[key] = cached_path
                return cached_path
        return self.get_config_path(filename)

    def get_active_config_dir(self) -> Path:
        """Returns the directory backing the active symai configuration."""
        symai_key = self._canonical_key('symai.config.json')
        cached = self._active_paths.get(symai_key)
        if cached is not None:
            return cached.parent
        for legacy_key, cached_path in list(self._active_paths.items()):
            if isinstance(legacy_key, Path) and (
                legacy_key.name == symai_key or str(legacy_key) == symai_key
            ):
                self._active_paths.pop(legacy_key, None)
                self._active_paths[symai_key] = cached_path
                return cached_path.parent
        return self.config_dir

SYMAI_CONFIG = {}
SYMSH_CONFIG = {}
SYMSERVER_CONFIG = {}
HOME_PATH = Path(SymAIConfig().config_dir)
