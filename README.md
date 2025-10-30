# ChartGemma - Multimodal Chart Understanding System

A complete system for chart analysis using ChartGemma (PaliGemma-based) model, with support for inference via Streamlit app and model finetuning on ChartQA dataset.

## 📁 Project Structure

```
MMLM/
├── app.py                    # Main Streamlit application for inference
├── model_loader.py           # Model and processor loading utilities
├── predictor.py              # Prediction/inference logic
├── utils.py                  # Utility functions (SSL setup, image downloading)
├── train.py                  # Main training script for finetuning
├── dataset_loader.py         # ChartQA dataset loading and preprocessing
├── training_config.py        # Training hyperparameters configuration
├── evaluate_model.py         # Standalone model evaluation script
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── ChartQA Dataset/         # ChartQA dataset directory
│   ├── train/
│   ├── val/
│   └── test/
└── README.md                # This file
```

## 📄 File Descriptions

### Core Application Files

#### `app.py`
**Purpose**: Main Streamlit web application for chart analysis

**What it does**:
- Creates a web interface with two-column layout (input/output)
- Handles image uploads and example image selection
- Displays model predictions in real-time
- Manages session state for model loading (loads once, reuses across interactions)
- Shows device information (CPU/GPU) in sidebar

**Key Features**:
- Image upload support (PNG, JPG, JPEG)
- Pre-loaded example charts with default prompts
- Persistent output display
- Error handling with user-friendly messages

**Flow**:
1. Sets up environment (file watcher type)
2. Loads model on first run (cached in session state)
3. Downloads example images if needed
4. Renders UI components
5. On "Analyze Chart" click:
   - Validates inputs
   - Calls `predict()` function
   - Displays results

---

#### `model_loader.py`
**Purpose**: Centralized model loading and device management

**Functions**:
- `load_model()`: Loads ChartGemma model and processor from HuggingFace
- `setup_device()`: Determines and returns appropriate device (CPU/GPU)

**Why it exists**: 
- Keeps model loading logic separate and reusable
- Used by both `app.py` (inference) and `train.py` (training)

---

#### `predictor.py`
**Purpose**: Core inference logic for generating chart analysis

**What it does**:
- Takes PIL image and text prompt as input
- Processes image and text through the processor
- Runs model generation
- Returns decoded output text

**Key Function**: `predict(image, input_text, model, processor)`
- Converts image to RGB
- Tokenizes and processes inputs
- Generates text response (max 512 tokens)
- Returns cleaned output

**Used by**: `app.py` for real-time predictions

---

#### `utils.py`
**Purpose**: Utility functions for setup and data management

**Functions**:
- `setup_ssl()`: Configures SSL context for unverified certificates (needed for downloads)
- `download_example_images()`: Downloads example chart images from ChartQA repository

**Why needed**: 
- Example images are downloaded on-demand
- Handles SSL certificate verification issues
- Checks if images already exist before downloading

---

### Training Files

#### `train.py`
**Purpose**: Main script for finetuning ChartGemma on ChartQA dataset

**Components**:
1. **`ChartGemmaDataCollator`**: Custom data collator
   - Processes batches of images and text
   - Creates input sequences (prompt + answer)
   - Masks prompt tokens in labels (only predict answers)
   - Handles padding and tokenization

2. **`ChartGemmaEvaluator`**: Custom evaluation class
   - Generates actual text predictions (not just loss)
   - Computes exact-match accuracy
   - Compares predictions to ground truth
   - Returns metrics and sample predictions

3. **`main()`**: Training orchestration
   - Loads model and datasets
   - Configures training arguments
   - Runs training loop
   - Performs final evaluation
   - Saves best model

**Flow**:
1. Load configuration
2. Setup device (CPU/GPU)
3. Load base model from HuggingFace
4. Load train/validation datasets
5. Create data collator and evaluator
6. Initialize Trainer with HuggingFace Transformers
7. Train model (with validation every 500 steps)
8. Run final accuracy evaluation
9. Save finetuned model

---

#### `dataset_loader.py`
**Purpose**: Dataset loading and preprocessing for ChartQA

**Components**:
1. **`ChartQADataset`**: PyTorch Dataset class
   - Loads JSON files containing questions/answers
   - Loads corresponding chart images from PNG directory
   - Handles image loading errors gracefully
   - Returns structured data for training

2. **`load_datasets()`**: Helper function
   - Loads train and/or validation datasets
   - Validates file existence
   - Returns dictionary with datasets

**Data Format**:
- Input: JSON files with `{"imgname": "...", "query": "...", "label": "..."}`
- Images: PNG files in corresponding `png/` directory
- Output: Dataset items with image, prompt text, and target answer

---

#### `training_config.py`
**Purpose**: Centralized training configuration

**What it provides**:
- `TrainingConfig` dataclass with all hyperparameters
- Default values for training settings
- Easy customization without editing code

**Key Parameters**:
- Model paths (base model, output directory)
- Training hyperparameters (epochs, batch size, learning rate)
- Generation settings (max length, max new tokens)
- Evaluation settings (eval steps, strategy)
- Optimizer settings (Adam parameters, weight decay)

**Usage**: Import and instantiate, or modify defaults

---

#### `evaluate_model.py`
**Purpose**: Standalone script for evaluating trained models

**What it does**:
- Loads a trained model from disk
- Evaluates on validation or test set
- Generates predictions and computes accuracy
- Displays detailed results with sample predictions

**Usage**:
```bash
python evaluate_model.py --model_path ./chartgemma-finetuned --split val
```

---

### Configuration Files

#### `requirements.txt`
All Python dependencies needed for the project:
- `streamlit`: Web app framework
- `torch`: PyTorch for deep learning
- `transformers`: HuggingFace transformers library
- `accelerate`: For distributed training
- `datasets`: Dataset utilities
- `pillow`: Image processing
- Other model-specific dependencies

#### `.streamlit/config.toml`
Streamlit configuration:
- Sets file watcher to polling mode (avoids PyTorch introspection issues)
- Configures logging levels

---

## 🔄 Complete System Flow

### Inference Flow (Streamlit App)

```
User Interaction
    ↓
app.py (Streamlit)
    ↓
1. Check session state → Load model if needed
    ↓
2. User uploads image or selects example
    ↓
3. User enters prompt/question
    ↓
4. Clicks "Analyze Chart" button
    ↓
predictor.py → predict()
    ↓
    a. Image preprocessing (RGB conversion)
    b. Text + Image → processor (tokenization)
    c. Inputs moved to device (CPU/GPU)
    d. Model generation (max 512 tokens)
    e. Decode output text
    ↓
5. Display results in UI
```

**Data Flow**:
1. **Input**: PIL Image + Text String
2. **Processing**: Processor converts to tensor format
3. **Model**: PaliGemmaForConditionalGeneration generates tokens
4. **Output**: Decoded text string with chart analysis

---

### Training Flow

```
train.py → main()
    ↓
1. Load TrainingConfig (hyperparameters)
    ↓
2. Setup device (CPU/GPU detection)
    ↓
3. Load base model from HuggingFace
    ↓
4. Load datasets (dataset_loader.py)
    ├── Load JSON files
    ├── Load images
    └── Create PyTorch Datasets
    ↓
5. Create data collator
    ├── Batch images and texts
    ├── Tokenize prompts + answers
    └── Mask prompt tokens in labels
    ↓
6. Initialize Trainer
    ├── Model
    ├── Training arguments
    ├── Train dataset
    ├── Validation dataset
    └── Data collator
    ↓
7. Training Loop (HuggingFace Trainer)
    ├── For each epoch:
    │   ├── For each batch:
    │   │   ├── Forward pass
    │   │   ├── Compute loss (only on answer tokens)
    │   │   ├── Backward pass
    │   │   └── Update weights
    │   └── Validation (every 500 steps)
    │       └── Compute validation loss
    └── Save checkpoints
    ↓
8. Final Evaluation (ChartGemmaEvaluator)
    ├── Generate predictions on validation set
    ├── Compute exact-match accuracy
    └── Display results
    ↓
9. Save best model
    ├── Model weights
    ├── Processor/tokenizer
    └── Config files
```

**Training Data Flow**:
1. **JSON Files**: Load question/answer pairs
2. **Images**: Load corresponding chart PNG files
3. **Batching**: ChartGemmaDataCollator combines image + prompt + answer
4. **Tokenization**: Processor creates input_ids with prompt+answer
5. **Labels**: Masked to only predict answer tokens
6. **Loss**: Computed only on answer prediction
7. **Optimization**: Adam optimizer updates model weights

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Inference App

```bash
# Start Streamlit app
streamlit run app.py
```

The app will:
1. Load the ChartGemma model (first time only, ~30 seconds)
2. Download example images if needed
3. Open browser at `http://localhost:8501`

### Finetuning the Model

```bash
# Run training (uses default config)
python train.py
```

Or customize in `training_config.py`:
```python
config = TrainingConfig(
    num_epochs=5,
    batch_size=8,
    learning_rate=1e-5
)
```

### Evaluating a Trained Model

```bash
# Evaluate on validation set
python evaluate_model.py --model_path ./chartgemma-finetuned --split val

# Evaluate on test set
python evaluate_model.py --model_path ./chartgemma-finetuned --split test
```

---

## 🏗️ Architecture Overview

### Model Architecture

**ChartGemma** (based on PaliGemma):
- Vision Encoder: Processes chart images
- Language Model: Generates text responses
- Multimodal Fusion: Combines vision and language

**Input**: Chart Image (PNG/JPG) + Text Question
**Output**: Text Answer

### Module Dependencies

```
app.py
├── model_loader.py ──┐
├── predictor.py      │
│   └── model_loader.py ─┐
└── utils.py          │  │
                      │  │
train.py ─────────────┘  │
├── model_loader.py ─────┘
├── dataset_loader.py
├── training_config.py
└── train.py (ChartGemmaEvaluator)

evaluate_model.py
├── model_loader.py
├── dataset_loader.py
└── train.py (ChartGemmaEvaluator, ChartGemmaDataCollator)
```

### Data Flow Diagram

```
┌─────────────────┐
│  User Input     │
│  (Image + Text) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   app.py        │
│  (Streamlit UI) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  predictor.py   │ ◄────┤ model_loader │
│  (Inference)    │      │  (Model)     │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│  Model Output   │
│  (Text Answer)  │
└─────────────────┘

Training Flow:
┌─────────────────┐
│  ChartQA Data   │
│  (JSON + PNG)   │
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ dataset_loader.py│
│  (Data Prep)     │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   train.py      │◄─────┤ model_loader │
│  (Training)     │      │  (Model)     │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│ Finetuned Model │
│  (Saved)        │
└─────────────────┘
```

---

## 📊 Dataset Information

**ChartQA Dataset**:
- **Training**: ~7,398 samples (human questions)
- **Validation**: ~960 samples
- **Test**: ~1,509 samples

**Data Format**:
- Each sample has:
  - `imgname`: Image filename (PNG)
  - `query`: Question about the chart
  - `label`: Ground truth answer

**Supported Chart Types**:
- Bar charts
- Line charts
- Pie charts
- Multi-column charts
- Stacked charts

---

## ⚙️ Configuration

### Model Configuration

Located in `model_loader.py`:
- Base model: `"ahmed-masry/chartgemma"`
- Can be changed to finetuned model path after training

### Training Configuration

Located in `training_config.py`:
- All hyperparameters configurable
- Default values optimized for reasonable training

### App Configuration

Located in `.streamlit/config.toml`:
- File watcher settings
- Logging configuration

---

## 🔍 Key Features

### Inference Features
✅ Real-time chart analysis
✅ Image upload support
✅ Example charts with pre-filled prompts
✅ Device detection (CPU/GPU)
✅ Error handling and user feedback

### Training Features
✅ Automatic validation during training
✅ Best model checkpointing
✅ Final accuracy evaluation
✅ Configurable hyperparameters
✅ Progress logging

### Evaluation Features
✅ Exact-match accuracy computation
✅ Sample prediction display
✅ Supports val/test splits
✅ Standalone evaluation script

---

## 🐛 Troubleshooting

### Common Issues

**Model loading takes too long**
- First load downloads model from HuggingFace (~GB)
- Subsequent loads use cached version
- Consider using GPU for faster inference

**Out of memory during training**
- Reduce `batch_size` in `training_config.py`
- Increase `gradient_accumulation_steps`
- Use `fp16=True` for mixed precision

**Evaluation slow on CPU**
- Training/evaluation is much faster on GPU
- Consider using cloud GPU instances for training
- Reduce eval dataset size for testing

**Streamlit file watcher errors**
- Already handled in `.streamlit/config.toml`
- If persists, restart Streamlit

---

## 📚 Usage Examples

### Example 1: Quick Inference

```python
from model_loader import load_model
from predictor import predict
from PIL import Image

model, processor = load_model()
image = Image.open("chart.png")
answer = predict(image, "What is the highest value?", model, processor)
print(answer)
```

### Example 2: Custom Training

```python
from training_config import TrainingConfig

config = TrainingConfig(
    num_epochs=5,
    batch_size=8,
    learning_rate=1e-5,
    output_dir="./my-model"
)
# Then run: python train.py
```

### Example 3: Evaluate Custom Model

```bash
python evaluate_model.py \
  --model_path ./custom-model \
  --dataset_root "ChartQA Dataset" \
  --split test
```

---

## 🔗 External Dependencies

- **HuggingFace Transformers**: Model architecture and training utilities
- **PyTorch**: Deep learning framework
- **ChartGemma Model**: Pre-trained on `ahmed-masry/chartgemma`
- **ChartQA Dataset**: Used for finetuning (included in project)

---

## 📝 Notes

- **First Run**: Model download takes time (~30 seconds) and space (~GB)
- **GPU Recommended**: Training is much faster on GPU, inference works on CPU
- **Memory**: Model requires ~4-8GB RAM for inference, more for training
- **Validation**: Runs automatically during training, final accuracy computed after training

---

## 🎯 Next Steps

1. **Run inference**: Start with `streamlit run app.py`
2. **Finetune**: Adjust config and run `python train.py`
3. **Evaluate**: Use `evaluate_model.py` to test performance
4. **Customize**: Modify `training_config.py` for your use case

---

## 📄 License

Check the license of:
- ChartGemma model
- ChartQA dataset
- Base libraries (Transformers, PyTorch, etc.)

---

**Built with ❤️ for multimodal chart understanding**

