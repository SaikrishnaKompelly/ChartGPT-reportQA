"""Script to evaluate a trained ChartGemma model on validation/test data."""
import torch
import argparse
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from .dataset_loader import load_datasets, ChartQADataset
from .train import ChartGemmaDataCollator, ChartGemmaEvaluator
from src.models.chartgemma import setup_device


def evaluate_model(model_path: str, dataset_root: str, split: str = "val"):
	"""Evaluate a trained model on validation or test data.
	
	Args:
		model_path: Path to the trained model directory
		dataset_root: Root directory of ChartQA Dataset
		split: Which split to evaluate on ("val" or "test")
	"""
	device = setup_device()
	print(f"Using device: {device}")
	
	# Load model
	print(f"Loading model from: {model_path}")
	model = PaliGemmaForConditionalGeneration.from_pretrained(model_path)
	processor = AutoProcessor.from_pretrained(model_path)
	
	model.to(device)
	model.eval()
	
	# Load dataset
	if split == "val":
		json_path = f"{dataset_root}/val/val_human.json"
		image_dir = f"{dataset_root}/val/png"
	elif split == "test":
		json_path = f"{dataset_root}/test/test_human.json"
		image_dir = f"{dataset_root}/test/png"
	else:
		raise ValueError(f"Unknown split: {split}. Use 'val' or 'test'")
	
	print(f"Loading {split} dataset...")
	eval_dataset = ChartQADataset(
		json_path=json_path,
		image_dir=image_dir,
		processor=processor,
		max_length=512
	)
	
	print(f"Loaded {len(eval_dataset)} samples")
	
	# Create data collator and evaluator
	data_collator = ChartGemmaDataCollator(processor, max_length=512)
	evaluator = ChartGemmaEvaluator(
		processor=processor,
		device=device,
		max_new_tokens=128
	)
	
	# Run evaluation
	print("\n" + "="*50)
	print(f"Evaluating on {split} set...")
	print("="*50)
	
	metrics = evaluator.evaluate(
		model=model,
		eval_dataset=eval_dataset,
		data_collator=data_collator,
		batch_size=4
	)
	
	# Print results
	print(f"\n{'='*50}")
	print(f"Evaluation Results ({split.upper()} set):")
	print(f"{'='*50}")
	print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['exact_matches']}/{metrics['total']})")
	print(f"  Exact Matches: {metrics['exact_matches']}")
	print(f"  Total Samples: {metrics['total']}")
	
	print(f"\nSample Predictions (first 10):")
	print(f"{'Match':<6} {'Prediction':<50} {'Ground Truth':<50}")
	print("-" * 110)
	for pred, label in zip(metrics['predictions'], metrics['labels']):
		match = "✓" if pred.lower() == label.lower() else "✗"
		print(f"{match:<6} {pred[:48]:<50} {label[:48]:<50}")
	
	return metrics


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate a trained ChartGemma model")
	parser.add_argument(
		"--model_path",
		type=str,
		default="./chartgemma-finetuned",
		help="Path to the trained model directory"
	)
	parser.add_argument(
		"--dataset_root",
		type=str,
		default="ChartQA Dataset",
		help="Root directory of ChartQA Dataset"
	)
	parser.add_argument(
		"--split",
		type=str,
		default="val",
		choices=["val", "test"],
		help="Which dataset split to evaluate on (val or test)"
	)
	
	args = parser.parse_args()
	
	evaluate_model(
		model_path=args.model_path,
		dataset_root=args.dataset_root,
		split=args.split
	)

