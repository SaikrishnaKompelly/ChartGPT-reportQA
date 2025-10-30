"""Application configuration for model providers and defaults."""
from dataclasses import dataclass
from typing import Literal
from .utils.env import get_env

ProviderName = Literal["ChartGemma", "OpenAI GPT-4o", "Google Gemini"]


@dataclass
class AppConfig:
	"""Config defaults read from environment variables."""
	# Default provider: ChartGemma | OpenAI GPT-4o | Google Gemini
	default_provider: ProviderName = (get_env("DEFAULT_PROVIDER", "ChartGemma") or "ChartGemma")  # type: ignore
	
	# ChartGemma local/hub model
	chartgemma_model: str = get_env("CHARTGEMMA_MODEL", "ahmed-masry/chartgemma") or "ahmed-masry/chartgemma"
	
	# OpenAI model name
	openai_model: str = get_env("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
	
	# Google Gemini model name
	gemini_model: str = get_env("GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash"
