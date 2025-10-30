"""ChartGemma model loader (PaliGemma-based)"""
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


def load_model():
	model = PaliGemmaForConditionalGeneration.from_pretrained("ahmed-masry/chartgemma")
	processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")
	return model, processor


def setup_device():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")
