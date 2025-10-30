"""Unified predictor that routes to selected provider."""
from PIL import Image

from src.models.chartgemma import setup_device


def predict_chartgemma(image: Image.Image, input_text: str, model, processor) -> str:
	device = setup_device()
	model.to(device)

	image = image.convert("RGB")
	inputs = processor(text=input_text, images=image, return_tensors="pt")
	inputs = {k: v.to(device) for k, v in inputs.items()}

	prompt_length = inputs['input_ids'].shape[1]
	generate_ids = model.generate(**inputs, max_new_tokens=512)
	output_text = processor.batch_decode(
		generate_ids[:, prompt_length:],
		skip_special_tokens=True,
		clean_up_tokenization_spaces=False
	)[0]
	return output_text
