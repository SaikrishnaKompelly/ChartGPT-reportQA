"""Main training script for finetuning ChartGemma on ChartQA dataset."""
import os
import torch
from transformers import (
	AutoProcessor,
	PaliGemmaForConditionalGeneration,
	TrainingArguments,
	Trainer
)
from torch.utils.data import DataLoader, Dataset
from .dataset_loader import load_datasets
from .training_config import TrainingConfig
from src.models.chartgemma import setup_device
import json


class ChartGemmaDataCollator:
	"""Custom data collator for ChartGemma training."""
	
	def __init__(self, processor, max_length=512):
		self.processor = processor
		self.max_length = max_length
	
	def __call__(self, batch):
		"""Collate a batch of samples."""
		# Extract images, prompts, and targets
		images = []
		texts = []
		targets = []
		
		for item in batch:
			images.append(item['image'])
			texts.append(item['input_text'])
			targets.append(item['target_text'])
		
		# Create full sequences with prompt and target
		# Format: prompt + " " + answer (for training, we want to predict the answer)
		full_sequences = []
		for text, target in zip(texts, targets):
			# Format: prompt + " " + answer
			full_sequence = f"{text} {target}"
			full_sequences.append(full_sequence)
		
		# Process images and full sequences together
		processed = self.processor(
			text=full_sequences,
			images=images,
			return_tensors="pt",
			padding=True,
			max_length=self.max_length,
			truncation=True
		)
		
		# Create labels - we want to predict only the answer part
		# For PaliGemma/ChartGemma, we need to mask the prompt tokens
		# The processor adds special tokens, so we need to account for those
		
		# Tokenize prompts separately to find where they end in the full sequence
		labels = processed['input_ids'].clone()
		
		# Process prompts to find their lengths (with special tokens)
		prompt_inputs = self.processor.tokenizer(
			texts,
			return_tensors="pt",
			padding=True,
			max_length=self.max_length,
			truncation=True
		)
		prompt_lengths = prompt_inputs['attention_mask'].sum(dim=1).tolist()
		
		# Mask out prompt tokens and padding tokens (set to -100 which is ignored in loss)
		for i, prompt_len in enumerate(prompt_lengths):
			# Mask the prompt part (keep answer part for prediction)
			labels[i, :prompt_len] = -100
			# Also mask padding tokens
			pad_token_id = self.processor.tokenizer.pad_token_id
			if pad_token_id is not None:
				labels[i][labels[i] == pad_token_id] = -100
		
		return {
			'pixel_values': processed['pixel_values'],
			'input_ids': processed['input_ids'],
			'attention_mask': processed['attention_mask'],
			'labels': labels
		}


class ChartGemmaEvaluator:
	"""Evaluator for ChartGemma that generates actual predictions."""
	
	def __init__(self, processor, device, max_new_tokens=128):
		self.processor = processor
		self.device = device
		self.max_new_tokens = max_new_tokens
	
	def evaluate(self, model, eval_dataset, data_collator, batch_size=4):
		"""Evaluate the model by generating predictions."""
		model.eval()
		
		eval_dataloader = DataLoader(
			eval_dataset,
			batch_size=batch_size,
			collate_fn=data_collator,
			shuffle=False
		)
		
		all_predictions = []
		all_labels = []
		exact_matches = 0
		total = 0
		
		with torch.no_grad():
			for batch_idx, batch in enumerate(eval_dataloader):
				# Move batch to device
				pixel_values = batch['pixel_values'].to(self.device)
				input_ids = batch['input_ids'].to(self.device)
				attention_mask = batch['attention_mask'].to(self.device)
				
				# Get prompt length for each item
				# We need to find where the prompt ends to only generate the answer
				prompt_lengths = []
				for i in range(input_ids.shape[0]):
					# Find actual length (non-padding)
					actual_length = attention_mask[i].sum().item()
					prompt_lengths.append(actual_length)
				
				# Generate predictions
				try:
					generate_ids = model.generate(
						pixel_values=pixel_values,
						input_ids=input_ids,
						attention_mask=attention_mask,
						max_new_tokens=self.max_new_tokens,
						do_sample=False,
						num_beams=1,
						pad_token_id=self.processor.tokenizer.pad_token_id
					)
					
					# Extract only the generated tokens (new tokens)
					batch_start_idx = total
					for i, prompt_len in enumerate(prompt_lengths):
						# Get generated tokens (after prompt)
						generated_ids = generate_ids[i, prompt_len:]
						
						# Decode prediction
						pred_text = self.processor.tokenizer.decode(
							generated_ids,
							skip_special_tokens=True,
							clean_up_tokenization_spaces=False
						).strip()
						
						# Get ground truth label using correct index
						sample_idx = batch_start_idx + i
						true_label = eval_dataset[sample_idx]['target_text'].strip()
						
						all_predictions.append(pred_text)
						all_labels.append(true_label)
						
						# Check exact match (case-insensitive)
						if pred_text.lower() == true_label.lower():
							exact_matches += 1
						
						total += 1
						
				except Exception as e:
					print(f"Error generating predictions for batch {batch_idx}: {e}")
					# Add empty predictions for failed batches
					batch_size_actual = pixel_values.shape[0]
					for i in range(batch_size_actual):
						all_predictions.append("")
						all_labels.append(eval_dataset[total + i]['target_text'])
						total += 1
		
		accuracy = exact_matches / total if total > 0 else 0.0
		
		return {
			"accuracy": accuracy,
			"exact_matches": exact_matches,
			"total": total,
			"predictions": all_predictions[:10],  # Return first 10 for inspection
			"labels": all_labels[:10]
		}


def main():
	"""Main training function."""
	# Load configuration
	config = TrainingConfig()
	
	# Setup device
	device = setup_device()
	print(f"Using device: {device}")
	
	# Load model and processor
	print(f"Loading model: {config.model_name}")
	model = PaliGemmaForConditionalGeneration.from_pretrained(
		config.model_name,
		torch_dtype=torch.float16 if config.fp16 and torch.cuda.is_available() else torch.float32
	)
	processor = AutoProcessor.from_pretrained(config.model_name)
	
	# Move model to device
	model.to(device)
	model.train()
	
	# Load datasets
	print(f"Loading datasets from: {config.dataset_root}")
	datasets = load_datasets(
		config.dataset_root,
		processor,
		max_length=config.max_length,
		use_train=True,
		use_val=True
	)
	
	if 'train' not in datasets:
		raise ValueError("Training dataset not found!")
	
	train_dataset = datasets['train']
	eval_dataset = datasets.get('val', None)
	
	print(f"Training samples: {len(train_dataset)}")
	if eval_dataset:
		print(f"Validation samples: {len(eval_dataset)}")
	
	# Create data collator
	data_collator = ChartGemmaDataCollator(processor, max_length=config.max_length)
	
	# Setup training arguments
	# Prepare max_steps - only include if not None
	training_kwargs = {
		"output_dir": config.output_dir,
		"num_train_epochs": config.num_epochs,
		"per_device_train_batch_size": config.batch_size,
		"per_device_eval_batch_size": config.batch_size,
		"gradient_accumulation_steps": config.gradient_accumulation_steps,
		"learning_rate": config.learning_rate,
		"warmup_steps": config.warmup_steps,
		"logging_steps": config.logging_steps,
		"save_steps": config.save_steps,
		"eval_steps": config.eval_steps if eval_dataset else None,
		"eval_strategy": config.eval_strategy if eval_dataset else "no",
		"save_total_limit": config.save_total_limit,
		"load_best_model_at_end": True if eval_dataset else False,
		"metric_for_best_model": "loss",  # Use loss for best model selection
		"weight_decay": config.weight_decay,
		"adam_beta1": config.adam_beta1,
		"adam_beta2": config.adam_beta2,
		"adam_epsilon": config.adam_epsilon,
		"fp16": config.fp16 and torch.cuda.is_available(),
		"dataloader_num_workers": config.dataloader_num_workers,
		"remove_unused_columns": config.remove_unused_columns,
		"report_to": "none",  # Can be changed to "tensorboard" or "wandb"
	}
	
	# Only add max_steps if it's not None
	if config.max_steps is not None:
		training_kwargs["max_steps"] = config.max_steps
	
	training_args = TrainingArguments(**training_kwargs)
	
	# Create evaluator for validation
	evaluator = None
	if eval_dataset:
		evaluator = ChartGemmaEvaluator(
			processor=processor,
			device=device,
			max_new_tokens=config.max_new_tokens
		)
		print("Evaluation enabled - will validate during training")
	
	# Create trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=data_collator,
	)
	
	# Train
	print("Starting training...")
	trainer.train()
	
	# Run final evaluation if validation set exists
	if eval_dataset and evaluator:
		print("\n" + "="*50)
		print("Running final evaluation...")
		print("="*50)
		final_metrics = evaluator.evaluate(
			model=trainer.model,
			eval_dataset=eval_dataset,
			data_collator=data_collator,
			batch_size=config.batch_size
		)
		
		print(f"\nFinal Validation Results:")
		print(f"  Accuracy: {final_metrics['accuracy']:.4f} ({final_metrics['exact_matches']}/{final_metrics['total']})")
		print(f"\nSample Predictions (first 5):")
		for i, (pred, label) in enumerate(zip(final_metrics['predictions'][:5], final_metrics['labels'][:5])):
			match = "✓" if pred.lower() == label.lower() else "✗"
			print(f"  {match} Pred: '{pred}' | Label: '{label}'")
	
	# Save final model
	print(f"\nSaving model to {config.output_dir}")
	trainer.save_model()
	processor.save_pretrained(config.output_dir)
	
	print("Training completed!")


if __name__ == "__main__":
	main()

