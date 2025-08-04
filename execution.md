# Polygon Coloring UNet - Setup and Execution Guide

## Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended, but CPU training is possible)
- At least 8GB RAM (16GB recommended)
- 5-10GB free disk space

## Installation

### 1. Clone/Download the Project
```bash
# Create project directory
mkdir polygon-coloring-unet
cd polygon-coloring-unet

# Copy all the provided Python files to this directory:
# - unet_model.py
# - dataset_loader.py 
# - train.py
# - inference.ipynb
# - requirements.txt
```

### 2. Set up Python Environment
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download and Prepare Dataset
```bash
# Download the dataset from the provided Google Drive link
# https://drive.google.com/open?id=1QXLgo3ZfQPorGwhYVmZUEWO_sU3i1pHM

# Extract to project directory
unzip dataset.zip

# Your directory structure should look like:
# polygon-coloring-unet/
# ├── dataset/
# │   ├── training/
# │   │   ├── inputs/
# │   │   ├── outputs/
# │   │   └── data.json
# │   └── validation/
# │       ├── inputs/
# │       ├── outputs/
# │       └── data.json
# ├── unet_model.py
# ├── dataset_loader.py
# ├── train.py
# └── inference.ipynb
```

## Running the Project

### 1. Setup WandB (Weights & Biases)
```bash
# Login to WandB (create free account at https://wandb.ai)
wandb login
# Enter your API key when prompted
```

### 2. Training the Model

#### Quick Start (Default Parameters)
```bash
python train.py --data_dir dataset --wandb_project polygon-coloring-unet
```

#### Full Parameter Training
```bash
python train.py \
    --data_dir dataset \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --image_size 256 \
    --wandb_project polygon-coloring-unet
```

#### Training on Google Colab
```python
# In a Colab notebook cell:
!git clone your-repo-url  # or upload files manually
!pip install -r requirements.txt
!python train.py --data_dir dataset --batch_size 8 --epochs 50
```

#### Training Parameters Explained
- `--data_dir`: Path to dataset directory
- `--batch_size`: Batch size (reduce if GPU memory issues)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--image_size`: Input image size (256x256 recommended)
- `--wandb_project`: WandB project name for tracking

### 3. Monitoring Training
- Open your WandB dashboard to monitor training progress
- Check loss curves, sample predictions, and metrics
- Training typically takes 2-4 hours on T4 GPU for 100 epochs

### 4. Inference and Testing

#### Option A: Jupyter Notebook (Recommended)
```bash
# Start Jupyter
jupyter notebook

# Open inference.ipynb
# Update the checkpoint path in the notebook
# Run all cells to see inference results
```

#### Option B: Python Script Inference
```python
# Create a simple inference script
from unet_model import ConditionalUNet, ColorMapper
from dataset_loader import PolygonColorDataset
import torch
from PIL import Image

# Load trained model
checkpoint = torch.load('path/to/best_model.pth')
model = ConditionalUNet(n_channels=3, n_classes=3, num_colors=10)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference
# (See inference.ipynb for complete example)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
python train.py --batch_size 8  # or even 4

# Or train on CPU (much slower)
export CUDA_VISIBLE_DEVICES=""
python train.py --batch_size 4
```

#### 2. Dataset Not Found
```bash
# Ensure dataset structure is correct
ls dataset/training/inputs/  # Should show polygon images
ls dataset/training/outputs/ # Should show colored polygon images
ls dataset/training/data.json # Should exist
```

#### 3. WandB Issues
```bash
# Re-login to WandB
wandb login --relogin

# Or run without WandB (modify train.py to comment out wandb calls)
```

#### 4. Import Errors
```bash
# Ensure all files are in the same directory
# Check Python path
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
```

### Performance Optimization

#### For Limited GPU Memory:
- Reduce batch_size to 8 or 4
- Reduce image_size to 128
- Use gradient checkpointing (modify model)

#### For Faster Training:
- Use multiple GPUs with DataParallel
- Increase batch_size if memory allows
- Use mixed precision training (add to train.py)

## Expected Results

### Training Progress
- **Epochs 1-20**: Rapid loss decrease, basic color learning
- **Epochs 20-50**: Fine-tuning, improved shape fidelity
- **Epochs 50-100**: Convergence, minimal improvements

### Final Metrics (Expected)
- **Training Loss**: 0.02-0.05
- **Validation Loss**: 0.03-0.06
- **PSNR**: 28-35 dB
- **Visual Quality**: Good coloring with sharp edges

## File Descriptions

- **unet_model.py**: Contains the ConditionalUNet architecture
- **dataset_loader.py**: Dataset class and data loading utilities
- **train.py**: Main training script with WandB integration
- **inference.ipynb**: Jupyter notebook for model testing and inference
- **requirements.txt**: Python dependencies

## Next Steps After Training

1. **Evaluate Results**: Use inference.ipynb to test model performance
2. **Experiment**: Try different hyperparameters or architectural changes
3. **Deploy**: Save best model for production use
4. **Enhance**: Add more colors, shapes, or advanced features

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all file paths and directory structure
3. Check GPU memory usage and reduce batch size if needed
4. Review WandB logs for training insights
5. Test with synthetic data first (as shown in inference.ipynb)

## Hardware Recommendations

### Minimum Requirements:
- **GPU**: Google Colab T4 (free)
- **RAM**: 8GB
- **Storage**: 5GB

### Recommended Setup:
- **GPU**: RTX 3080/4080 or V100
- **RAM**: 16GB+
- **Storage**: 10GB+ SSD

### Cloud Options:
- **Google Colab**: Free T4 GPU (limited hours)
- **Kaggle Notebooks**: Free GPU access
- **AWS/GCP**: On-demand GPU instances
