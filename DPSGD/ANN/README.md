# ANN Model Training with DPSGD

This project implements Differentially Private Stochastic Gradient Descent (DPSGD) for training Artificial Neural Networks (ANNs) on various datasets. It uses the Opacus library to ensure privacy-preserving training.

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
1. **`train_dpsgd.py`**: Main script to train models and run DPSGD.
2. **`utils.py`**: Utility functions for dataset loading and preprocessing.



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
- `scikit-learn`

To install the required libraries, run:
```bash
pip install -r requirements.txt
```

---

## Usage

### Training ANN Models and Performing MIA

Run the `train_dpsgd.py` script using the following arguments:

#### Arguments
- `--dataset`: Dataset to use (`cifar10`, `cifar100`, `mnist`, `fmnist`, `iris`, `breast_cancer`).
- `--num_epochs`: Number of epochs for training.
- `--batch_size`: Batch size for training.
- `--epsilon`: Privacy budget for DPSGD
- `--lr': Learning rate for the optimizer

#### Example Command
```bash
python train_dpsgd.py --dataset cifar10 --num_epochs 20 --batch_size 128 --epsilon 2.0 --lr 0.001 
```

---



## Model Details

### Convolutional Neural Network (ANNConvNet)
Used for image datasets (`cifar10`, `cifar100`, `mnist`, `fmnist`):
- Two convolutional layers with ReLU activation.
- Two fully connected layers for classification.

### Fully Connected Network (ANNFCNet)
Used for tabular datasets (`iris`, `breast_cancer`):
- Two fully connected layers with ReLU activation.

---

## Differentially Private Stochastic Gradient Descent (DPSGD)

The DPSGD module ensures that the model training process is privacy-preserving by adding controlled noise to gradients during optimization. This method protects individual data points in the training set from being exposed through model outputs:

- **Privacy Budget (ε)**: A fixed privacy budget is set to control the trade-off between accuracy and privacy.
- **Noise Addition**: Gaussian noise is added to gradients, scaled according to the `max_grad_norm` parameter.
- **Delta (δ)**: A small probability parameter for privacy guarantees.
- **Privacy Accounting**: The Opacus library's privacy accountant is used to track and report the accumulated privacy loss (ε) over epochs.

The final privacy budget (ε) is printed during training, along with the model's performance metrics such as loss and accuracy.
