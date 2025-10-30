"""Environment utilities for loading API keys and configuration."""
import os
from typing import Optional

try:
	from dotenv import load_dotenv
	_DOTENV_AVAILABLE = True
except Exception:
	_DOTENV_AVAILABLE = False


def load_environment(dotenv_path: Optional[str] = None) -> None:
	"""Load environment variables from a .env file if python-dotenv is installed.

	Args:
		dotenv_path: Optional custom path to .env file.
	"""
	if _DOTENV_AVAILABLE:
		if dotenv_path:
			load_dotenv(dotenv_path)
		else:
			load_dotenv()


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
	"""Get an environment variable with optional default.

	Args:
		key: Environment variable name.
		default: Default value if not set.

	Returns:
		The value or default if missing.
	"""
	return os.getenv(key, default)
