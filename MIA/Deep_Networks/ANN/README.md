
# README.md

## Advanced Neural Networks Implementation for Membership Inference Attacks (MIA)

This repository contains implementations of advanced neural network architectures, including ResNet and VGG16, for evaluating their privacy resilience against Membership Inference Attacks (MIA). The code is modular and supports multiple datasets with different architectures.

---

## Datasets Supported

### 1. CIFAR-10
- **Description**: 60,000 32x32 color images in 10 classes.
- **Models**: ResNet (6, 18, 50) and VGG16.

### 2. CIFAR-100
- **Description**: 60,000 32x32 color images in 100 classes.
- **Models**: ResNet (6, 18, 50) and VGG16.

### 3. MNIST
- **Description**: 70,000 grayscale 28x28 images in 10 classes (handwritten digits).
- **Models**: ResNet (6, 18, 50) and VGG16.

### 4. Fashion-MNIST (FMNIST)
- **Description**: 70,000 grayscale 28x28 images in 10 classes (fashion categories).
- **Models**: ResNet (6, 18, 50) and VGG16.

---

## Code Structure

### Files
1. **`train_mia.py`**: Main training and Membership Inference Attack (MIA) script.
2. **`model.py`**: Contains definitions for ResNet (6, 18, 50) and VGG16 architectures.
3. **`utils.py`**: Utility functions for loading datasets and preprocessing.

---

## Prerequisites

### Required Libraries
- `pytorch`
- `torchvision`
- `numpy`
- `pandas`
- `tqdm`
- `scikit-learn`
- `matplotlib`
- `opacus`
- `argparse`
- `pandas`
- `seaborn`
- `pickle`
- `hyperopt`
- `logging`
- `multiprocessing`

To install the required libraries, run:
```bash
pip install -r requirements.txt
```

---

## Usage

### Training Models and Performing MIA

Run the `train_mia.py` script using the following arguments:

#### Arguments
- `--dataset`: Dataset to use (`cifar10`, `cifar100`, `mnist`, `fmnist`).
- `--num_epochs`: Number of epochs for training.
- `--model`: Model architecture (`resnet6`, `resnet18`, `resnet50`, `vgg16`).

#### Example Command
```bash
python train_mia.py --dataset cifar10 --num_epochs 20 --model resnet18
```

---

## Directory Structure

After running the code, the directory structure will look as follows:

```bash
roc_data1/
|-- cifar10/
|   |-- class_0_roc_data.csv
|   |-- class_0_roc_curve.png
|   |-- ...
|-- ...
```

---

## Outputs

### Results
- **Per-Class AUC**: Printed for each class during the MIA process.
- **ROC Data**: Saved as `.csv` files for each class.
- **ROC Plots**: Saved as `.png` files for each class.

---

## Model Details

### ResNet Architectures
- **ResNet6**: Small version of ResNet for quick experiments.
- **ResNet18**: A standard deep ResNet with 18 layers.
- **ResNet50**: A deeper ResNet architecture for complex tasks.
- **Features**:
  - Uses `GroupNorm` for normalization.
  - Adjustable depth and capacity.

### VGG16
- A 16-layer deep network.
- Features:
  - Fully connected layers with `GroupNorm`.
  - Suitable for classification tasks with detailed features.

---

## Membership Inference Attack (MIA)

The MIA module evaluates the privacy of trained models by attacking them with SVM-based classifiers:
- **Shadow Model**: Simulates the behavior of the target model.
- **Target Model**: The main model being attacked.
- **ROC Data and AUC**: Used to assess attack success.

---

## Contact
For questions or issues, feel free to contact the repository maintainer.
