"""Utilities for accessing local ChartQA examples."""
import os
import json
from pathlib import Path
from typing import List, Dict


def get_local_examples(dataset_root: str = "ChartQA Dataset", split: str = "val", max_examples: int = 3) -> List[Dict[str, str]]:
	"""Return a few local examples (image path + prompt) from ChartQA Dataset.
	
	Args:
		dataset_root: Root path to ChartQA Dataset
		split: One of "train", "val", "test"
		max_examples: Number of examples to return
	
	Returns:
		List of dicts: [{"imgname", "image_path", "query", "label"}, ...]
	"""
	root = Path(dataset_root)
	json_file = root / split / f"{split}_human.json"
	png_dir = root / split / "png"
	if not json_file.exists() or not png_dir.exists():
		return []
	try:
		with open(json_file, "r") as f:
			items = json.load(f)
	except Exception:
		return []
	examples: List[Dict[str, str]] = []
	for item in items[:max_examples]:
		imgname = item.get("imgname")
		query = item.get("query", "")
		label = item.get("label", "")
		image_path = str(png_dir / imgname) if imgname else None
		if image_path and os.path.exists(image_path):
			examples.append({
				"imgname": imgname or "",
				"image_path": image_path,
				"query": query,
				"label": label,
			})
		if len(examples) >= max_examples:
			break
	return examples
