# Artificial Neural Network (ANN) Implementation for Membership Inference Attacks (MIA)

This repository contains an implementation of Artificial Neural Networks (ANNs) for evaluating their privacy resilience against Membership Inference Attacks (MIA). The code supports multiple datasets and model architectures.

---

## Datasets Supported

### 1. CIFAR-10
- **Description**: 60,000 32x32 color images in 10 classes.
- **Model**: Convolutional Neural Network (ConvNet).

### 2. CIFAR-100
- **Description**: 60,000 32x32 color images in 100 classes.
- **Model**: Convolutional Neural Network (ConvNet).

### 3. MNIST
- **Description**: 70,000 grayscale 28x28 images in 10 classes (handwritten digits).
- **Model**: Convolutional Neural Network (ConvNet).

### 4. Fashion-MNIST (FMNIST)
- **Description**: 70,000 grayscale 28x28 images in 10 classes (fashion categories).
- **Model**: Convolutional Neural Network (ConvNet).

### 5. Iris
- **Description**: A small dataset containing 150 samples of 4 features each, classified into 3 categories.
- **Model**: Fully Connected Network (FCNet).

### 6. Breast Cancer
- **Description**: A small dataset containing 569 samples of 30 features each, classified into 2 categories.
- **Model**: Fully Connected Network (FCNet).

---

## Code Structure

### Files
1. **`train.py`**: Main script to train models and run Membership Inference Attacks (MIA).
2. **`utils.py`**: Utility functions for dataset loading and preprocessing.

### Key Functions in `train.py`
- **`train_and_evaluate_model`**: Trains and evaluates ANN models for the given dataset.
- **`mia_attack`**: Executes the Membership Inference Attack.
- **`save_roc_data`**: Saves and plots ROC data for each class.

---

## Required Libraries

- `torch`
- `torchvision`
- `tqdm`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `argparse`
- `csv`
- `opacus`
- `argparse`
- `pandas`
- `seaborn`

To install the required libraries, run:
```bash
pip install -r requirements.txt
```

---

## Usage

### Training ANN Models and Performing MIA

Run the `train.py` script using the following arguments:

#### Arguments
- `--dataset`: Dataset to use (`cifar10`, `cifar100`, `mnist`, `fmnist`, `iris`, `breast_cancer`).
- `--num_epochs`: Number of epochs for training.
- `--batch_size`: Batch size for training.
- `--data_path`: Directory to save/download datasets.
- `--output_dir`: Directory to save models and results.

#### Example Command
```bash
python train.py --dataset cifar10 --num_epochs 20 --batch_size 128 --data_path ./data --output_dir ./output
```

---

## Directory Structure

After running the code, the directory structure will look as follows:

```bash
output/
|-- cifar10/
|   |-- run_1/
|       |-- roc_data/
|           |-- roc_model_0.csv
|           |-- roc_curve_model_0.png
|           |-- ...
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

### Convolutional Neural Network (ConvNet)
Used for image datasets (`cifar10`, `cifar100`, `mnist`, `fmnist`):
- Two convolutional layers with ReLU activation.
- Two fully connected layers for classification.

### Fully Connected Network (FCNet)
Used for tabular datasets (`iris`, `breast_cancer`):
- Two fully connected layers with ReLU activation.

---

## Membership Inference Attack (MIA)

The MIA module evaluates the privacy of trained models by attacking them with SVM-based classifiers:

- **Shadow Model**: A model trained on a subset of the training dataset.
- **Target Model**: The main model being attacked.
- **ROC Data and AUC**: Used to assess the attack success.
