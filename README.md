# DarkVision

DarkVision is a deep learning project focused on image classification using PyTorch and ResNet18 architecture. The project implements various loss functions and provides tools for training and testing image classification models.

## Project Structure

```
DarkVision/
├── dataset/                 # Directory containing training and validation datasets
├── model/                   # Directory for model-related files
├── final_results_NV/        # Directory for storing final results
├── checkpoint.pth           # Saved model checkpoint
├── training_output.txt      # Training process output
├── utils.py                 # Utility functions and data loading
├── test.py                  # Testing script
├── test_base.py             # Base testing functionality
├── train.py                 # Training script
├── loss.py                  # Custom loss function implementations
├── model.py                 # Model architecture definition
├── data.ipynb               # Jupyter notebook for data analysis
└── LICENSE                  # Project license file
```

## Features

- ResNet18-based image classification
- Multiple loss function options:
  - Focal Loss
  - Mean Squared Error (MSE)
  - Cross Entropy Loss
- Custom dataset handling
- Training and validation pipeline
- Model checkpointing
- GPU support (if available)

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- pandas
- numpy
- PIL (Python Imaging Library)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/michaelborek/DarkVision.git
cd DarkVision
```

2. Install the required dependencies:
```bash
pip install torch torchvision pandas numpy pillow
```

## Dataset Preparation

The project expects datasets in CSV format with the following structure:
- `filepath`: Path to the image file
- `label`: Class label for the image

Place your training and validation datasets in the `dataset/` directory:
- `train_dataset.csv`
- `val_dataset.csv`

## Usage

### Training

To train the model, use the following command:

```bash
python train.py --loss [loss_type]
```

Available loss types:
- `focal`: Focal Loss
- `mse`: Mean Squared Error
- `cross_entropy` or `ce`: Cross Entropy Loss

Example:
```bash
python train.py --loss focal
```

### Testing

To test the model, use:

```bash
python test.py
```

## Model Architecture

The project uses a ResNet18 architecture with the following modifications:
- Pretrained weights from ImageNet
- Customizable number of output classes
- Modified final fully connected layer

## Training Parameters

- Number of epochs: 8
- Batch size: 128
- Learning rate: 0.001
- Optimizer: Adam

## Results

Training progress and results are saved in:
- `training_output.txt`: Training metrics and progress
- `checkpoint.pth`: Model weights and training state
- `final_results_NV/`: Directory containing final evaluation results

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For any questions or suggestions, please open an issue in the repository.
