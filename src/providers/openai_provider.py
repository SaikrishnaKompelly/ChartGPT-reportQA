"""OpenAI GPT-4o provider for image+text chart QA."""
from typing import Optional
import base64
from io import BytesIO
from PIL import Image

from src.utils.env import get_env

try:
	from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
	OpenAI = None  # type: ignore


def _pil_to_base64_png(image: Image.Image) -> str:
	buf = BytesIO()
	image.save(buf, format="PNG")
	return base64.b64encode(buf.getvalue()).decode("utf-8")


def analyze_chart_with_openai(image: Image.Image, prompt: str, model: str = "gpt-4o-mini") -> str:
	"""Call OpenAI GPT-4o to analyze a chart.

	Args:
		image: PIL Image of the chart.
		prompt: User prompt/question.
		model: OpenAI multimodal model name.

	Returns:
		Generated answer text.
	"""
	api_key = get_env("OPENAI_API_KEY") or get_env("OPENROUTER_API_KEY")
	if OpenAI is None or not api_key:
		raise RuntimeError("OpenAI client not available or API key missing.")

	client = OpenAI(api_key=api_key)

	image_b64 = _pil_to_base64_png(image.convert("RGB"))

	completion = client.chat.completions.create(
		model=model,
		messages=[
			{
				"role": "user",
				"content": [
					{"type": "text", "text": prompt},
					{
						"type": "image_url",
						"image_url": {"url": f"data:image/png;base64,{image_b64}"},
					},
				],
			}
		],
	)
	return completion.choices[0].message.content or ""
