
# SNN Membership Inference Attack Project

# Spiking Neural Network for Membership Inference Attacks (MIA)

This project explores the application of spiking neural networks (SNNs) in assessing membership inference attack vulnerabilities. Specifically, we evaluate two neural network architectures:
- `gSResnet`: A modified spiking residual network.
- `SpikingVGG16`: A spiking neural network adaptation of the VGG16 architecture.

The experiments were conducted on datasets including CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST.



## Directory Structure

```plaintext
.
├── model.py            # Contains SNN architectures (gSResnet and SpikingVGG16)
├── train_mia.py        # Training script for target and shadow models
├── utils.py            # Helper functions for training and evaluation
├── data/               # Directory to store datasets
├── models/             # Directory to save trained model weights
├── roc_data_v/         # Directory to store ROC curve data and AUC plots
```

---


## Requirements

To set up the environment, install the required packages:

```bash
pip install torch torchvision matplotlib scikit-learn pandas snntorch
```

## Dataset Preparation

The following datasets are supported in this project:
- CIFAR-10
- CIFAR-100
- MNIST
- Fashion-MNIST (FMNIST)

The datasets will be automatically downloaded to the specified `data` folder when running the scripts.

## Training and Evaluating the Models

### Train the Target and Shadow Models

To train the models and perform the MIA, run:

```bash
python train_mia.py --dataset cifar10 --arch gsresnet --num_steps 50 --num_epochs 70 --batch_size 128 --lr 0.0268
```

#### Example:
```bash
python train_mia.py --dataset cifar10 --arch spikingvgg16 --num_steps 50 --num_epochs 70 --batch_size 128 --lr 0.0268
```

### Key Arguments:

- `--dataset`: The dataset to use. Options: `cifar10`, `cifar100`, `mnist`, `fmnist`, 
- `--arch`: The SNN architecture. Options: `sresnet`, `gsresnet`, `spikingvgg16`.
- `--num_steps`: Number of time steps for SNN simulation.
- `--num_epochs`: Number of training epochs.

### Custom Configurations

Additional parameters can be adjusted:
- `--lr`: Learning rate for training (default: `0.0268`).
- `--leak_mem`: Membrane leakage rate (default: `0.874`).
- `--batch_size`: Batch size for training (default: `21`).


### Parameters
- `--dataset`: Dataset to use. Options: `cifar10`, `cifar100`, `mnist`, `fmnist`, `cifar10dvs`.
- `--arch`: Architecture. Options: `gsresnet`, `spikingvgg16`.
- `--num_steps`: Number of simulation timesteps for the SNN.
- `--num_epochs`: Number of training epochs.
- `--batch_size`: Batch size for training and evaluation.
- `--lr`: Learning rate for training.

---

## Results

### Membership Inference Attacks
The membership inference attacks were conducted using shadow models trained with the same architecture as the target models. The results were evaluated using ROC curves and AUC scores.

The ROC curves and AUC scores for each class are saved in the `roc_data_v/` directory. For example, the following files are generated:
- `class_0_roc_data.csv`
- `class_0_roc_curve.png`

### Models
- `gSResnet`: Demonstrates robustness due to its residual connections and spiking dynamics.
- `SpikingVGG16`: Offers a deeper network structure with spiking-based computations.

---

## Notes
- All SNN layers use surrogate gradients for backpropagation.
- Membrane potential leakage (`leak_mem`) and other hyperparameters can be adjusted via the command-line arguments.

For further customization or questions, please refer to the code comments in `train_mia.py` and `model.py`.


## Results

The project generates the following outputs:

- **ROC Data**: Saved in the `roc_data_v/<dataset>/` directory.
  - Example: `roc_data_v/cifar10/class_0_roc_data.csv`
- **ROC Curves**: Plotted for each class and saved as PNG images.
  - Example: `roc_data_v/cifar10/class_0_roc_curve.png`
- **Model Checkpoints**: The best model weights are saved in the `models/` directory.

### Visualizing ROC Curves

The generated ROC curves for each class can be found in the `roc_data_v` directory.


