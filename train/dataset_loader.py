"""Dataset loader for ChartQA finetuning."""
import json
import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional


class ChartQADataset(Dataset):
	"""Dataset class for loading ChartQA data for finetuning ChartGemma."""
	
	def __init__(
		self, 
		json_path: str, 
		image_dir: str, 
		processor,
		max_length: int = 512
	):
		"""Initialize the ChartQA dataset.
		
		Args:
			json_path: Path to the JSON file with questions and answers
			image_dir: Directory containing the chart images
			processor: The model processor for tokenization
			max_length: Maximum sequence length for generation
		"""
		self.processor = processor
		self.max_length = max_length
		self.image_dir = Path(image_dir)
		
		# Load JSON data
		with open(json_path, 'r') as f:
			self.data = json.load(f)
		
		print(f"Loaded {len(self.data)} samples from {json_path}")
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		"""Get a single training example."""
		item = self.data[idx]
		
		# Load image
		img_path = self.image_dir / item['imgname']
		try:
			image = Image.open(img_path).convert("RGB")
		except Exception as e:
			# If image can't be loaded, return None and handle in collator
			print(f"Warning: Could not load image {img_path}: {e}")
			# Create a blank image as fallback
			image = Image.new("RGB", (224, 224), color="white")
		
		# Format the prompt - ChartGemma uses a specific format
		# We use the query as the prompt and label as the target
		prompt = item['query']
		target = item['label']
		
		# Store raw data - processing will happen in data collator
		return {
			'image': image,
			'input_text': prompt,
			'target_text': target
		}


def load_datasets(
	dataset_root: str,
	processor,
	max_length: int = 512,
	use_train: bool = True,
	use_val: bool = True
):
	"""Load train and validation datasets.
	
	Args:
		dataset_root: Root directory of ChartQA Dataset
		processor: Model processor
		max_length: Maximum sequence length
		use_train: Whether to load training set
		use_val: Whether to load validation set
	
	Returns:
		Dictionary with 'train' and/or 'val' datasets
	"""
	datasets = {}
	dataset_root = Path(dataset_root)
	
	if use_train:
		train_json = dataset_root / "train" / "train_human.json"
		train_images = dataset_root / "train" / "png"
		if train_json.exists() and train_images.exists():
			datasets['train'] = ChartQADataset(
				str(train_json),
				str(train_images),
				processor,
				max_length
			)
	
	if use_val:
		val_json = dataset_root / "val" / "val_human.json"
		val_images = dataset_root / "val" / "png"
		if val_json.exists() and val_images.exists():
			datasets['val'] = ChartQADataset(
				str(val_json),
				str(val_images),
				processor,
				max_length
			)
	
	return datasets

