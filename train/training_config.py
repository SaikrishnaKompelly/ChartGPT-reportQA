"""Training configuration for ChartGemma finetuning."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
	"""Configuration class for training parameters."""
	
	# Data paths
	dataset_root: str = "ChartQA Dataset"
	output_dir: str = "./chartgemma-finetuned"
	
	# Model
	model_name: str = "ahmed-masry/chartgemma"
	
	# Training hyperparameters
	num_epochs: int = 3
	batch_size: int = 4
	gradient_accumulation_steps: int = 4
	learning_rate: float = 2e-5
	warmup_steps: int = 500
	max_steps: Optional[int] = None
	
	# Generation parameters
	max_length: int = 512
	max_new_tokens: int = 128
	
	# Training settings
	save_steps: int = 500
	eval_steps: int = 500
	logging_steps: int = 100
	save_total_limit: int = 3
	
	# Optimizer
	weight_decay: float = 0.01
	adam_beta1: float = 0.9
	adam_beta2: float = 0.999
	adam_epsilon: float = 1e-8
	
	# Other
	fp16: bool = True  # Use mixed precision training if GPU available
	dataloader_num_workers: int = 4
	remove_unused_columns: bool = False
	
	# Evaluation
	eval_strategy: str = "steps"  # "steps" or "epoch"
	
	def __post_init__(self):
		"""Validate configuration."""
		if self.max_steps is not None and self.max_steps <= 0:
			raise ValueError("max_steps must be positive or None")

