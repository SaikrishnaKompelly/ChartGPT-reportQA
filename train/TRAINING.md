# ChartGemma Finetuning Guide (train/)

This guide explains how to finetune the ChartGemma model on your ChartQA Dataset using the code in the `train/` package.

## Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Verify your dataset structure:
   ```
   ChartQA Dataset/
   ├── train/
   │   ├── png/
   │   └── train_human.json
   ├── val/
   │   ├── png/
   │   └── val_human.json
   └── test/
       ├── png/
       └── test_human.json
   ```

## Quick Start

### Basic Training

Run the training script:
```bash
a) python -m train.train
# or
b) python train/train.py
```

This will:
- Load the ChartGemma model (`ahmed-masry/chartgemma`)
- Load training and validation data from `ChartQA Dataset/`
- Train for 3 epochs with default hyperparameters
- Save the finetuned model to `./chartgemma-finetuned/`

### Customizing Training

Edit `train/training_config.py` to adjust hyperparameters:
```python
@dataclass
class TrainingConfig:
    dataset_root: str = "ChartQA Dataset"
    output_dir: str = "./chartgemma-finetuned"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    # ... more options
```

Or override in code where `TrainingConfig()` is instantiated.

## Training Parameters

- `num_epochs`: Number of training epochs (default: 3)
- `batch_size`: Batch size per device (default: 4)
- `learning_rate`: Learning rate (default: 2e-5)
- `max_length`: Maximum sequence length (default: 512)
- `gradient_accumulation_steps`: Accumulate gradients over steps (default: 4)
- `fp16`: Use mixed precision training (default: True if GPU available)

## Using the Finetuned Model

After training, use the saved model in your app:
```python
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
model = PaliGemmaForConditionalGeneration.from_pretrained("./chartgemma-finetuned")
processor = AutoProcessor.from_pretrained("./chartgemma-finetuned")
```

## Validation During Training

1. Automatic validation during training (every `eval_steps`, default 500):
   - Computes validation loss and saves best model
2. Final accuracy evaluation after training completes:
   - Generates predictions and computes exact-match accuracy
3. Standalone evaluation:
   ```bash
   python -m train.evaluate_model --model_path ./chartgemma-finetuned --split val
   ```

## Troubleshooting

- Out of memory: reduce `batch_size`, enable fp16, or reduce `max_length`
- Import errors: `pip install -r requirements.txt`
- Dataset not found: Check paths in `train/training_config.py`
- Slow training: use GPU if available, reduce workers
