# Neuromorphic Privacy: Exploring Privacy in Neuromorphic and Conventional Neural Architectures

This repository accompanies the study **"Are Neuromorphic Architectures Inherently Privacy-Preserving? An Exploratory Study"**, focusing on the comparative evaluation of Artificial Neural Networks (ANNs) and Spiking Neural Networks (SNNs) in terms of privacy preservation. The repository provides code, experiments, and results for evaluating Membership Inference Attacks (MIA) and Differentially Private Stochastic Gradient Descent (DPSGD) across various architectures and datasets.

## Overview

Neuromorphic computing, exemplified by Spiking Neural Networks (SNNs), offers advantages in energy efficiency and biological inspiration. However, their potential for privacy preservation remains underexplored. This repository examines:
1. **Privacy Vulnerabilities**: Comparing ANNs and SNNs under MIA.
2. **Differential Privacy**: Evaluating privacy-utility trade-offs with DPSGD.
3. **Learning Algorithms**: Assessing the impact of surrogate gradients and evolutionary algorithms on SNN privacy.

## Repository Structure

The repository is organized into key modules, each with dedicated code and results:

### `Baseline`
Contains baseline ANN and SNN implementations with simple architectures for privacy analysis.

- `ANN/`: ANN-specific implementations (e.g., ConvNet, FCNet).
- `SNN/`: SNN-specific implementations with surrogate gradient-based training.

### `Deep_Networks`
Advanced architectures such as ResNet18 and VGG16, implemented for both ANNs and SNNs.

- `ANN/`: Deep ANN implementations with batch normalization.
- `SNN/`: Deep SNN implementations using spiking neurons and temporal dynamics.

### `DPSGD`
Implements Differentially Private Stochastic Gradient Descent for ANN and SNN models.

- Contains utility scripts for evaluating the privacy-utility trade-off across datasets.

### `MIA`
Membership Inference Attack implementations with shadow and target models.

- Evaluates attack success using AUC and ROC metrics for both ANNs and SNNs.

### `snnenv`
Environment setup instructions and configuration for running SNN experiments using snnTorch and other frameworks.

### `requirements.txt`
Lists the dependencies required to run the experiments, including PyTorch, snnTorch, Opacus, and other libraries.

## Setup Instructions

To get started:
1. Clone this repository:
   ```
   git clone https://github.com/your-username/Neuromorphic_Privacy.git
   cd Neuromorphic_Privacy
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Navigate to the desired folder (e.g., `Baseline/ANN`) and run the scripts.

## How to Use

1. **Baseline Evaluation**:
   - Train baseline ANN/SNN models using `train.py` in the respective subfolder.
   - Evaluate membership inference attacks or apply DPSGD.

2. **Advanced Architectures**:
   - Experiment with deeper models (ResNet18, VGG16) in the `Deep_Networks` folder.

3. **Privacy Metrics**:
   - Use the `MIA` module to evaluate attack success.
   - Use the `DPSGD` module to analyze privacy-utility trade-offs.

## Datasets

Supported datasets include:
- Image datasets: MNIST, F-MNIST, CIFAR-10, CIFAR-100.
- Tabular datasets: Iris, Breast Cancer.

Data preprocessing is handled by scripts in `utils.py` across modules.

## Results

For details on experimental findings, refer to individual `README.md` files within each module. Key results include:
- Privacy resilience metrics (AUC scores).
- Comparative utility under DPSGD.

