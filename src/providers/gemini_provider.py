"""Google Gemini provider for image+text chart QA."""
from PIL import Image

from src.utils.env import get_env

try:
	import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
	genai = None  # type: ignore


def analyze_chart_with_gemini(image: Image.Image, prompt: str, model: str = "gemini-1.5-flash") -> str:
	"""Call Google Gemini to analyze a chart.

	Args:
		image: PIL Image of the chart.
		prompt: User prompt/question.
		model: Gemini multimodal model name.

	Returns:
		Generated answer text.
	"""
	api_key = get_env("GOOGLE_API_KEY")
	if genai is None or not api_key:
		raise RuntimeError("Gemini client not available or GOOGLE_API_KEY missing.")

	genai.configure(api_key=api_key)
	gm = genai.GenerativeModel(model)

	# Gemini supports passing PIL Images directly as a part
	response = gm.generate_content([prompt, image.convert("RGB")])
	return (response.text or "").strip()
