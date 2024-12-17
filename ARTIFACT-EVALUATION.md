# Artifact Appendix

Paper title: **Are Neuromorphic Architectures Inherently Privacy-preserving? An Exploratory Study**

Artifacts HotCRP Id: **#26**

Requested Badge: **Functional**

## Description
This artifact evaluates the functionality of the models and methods proposed in the paper. The key components include Membership Inference Attack (MIA) evaluations and Differentially Private Stochastic Gradient Descent (DPSGD) implementations for Spiking Neural Networks (SNNs) and Artificial Neural Networks (ANNs). These artifacts directly support the paper’s claims about the privacy-preserving properties of neuromorphic architectures under MIA and DPSGD.

### Security/Privacy Issues and Ethical Concerns
- The artifact does not require disabling any security mechanisms such as firewalls or ASLR.
- No malware or malicious code is included.
- Ethical considerations:
  - The datasets used are publicly available (CIFAR10, CIFAR100, MNIST, FashionMNIST, Iris, and Breast Cancer).
  - Ensure proper acknowledgment when using datasets or code.

## Basic Requirements

### Hardware Requirements
- **Minimal Hardware**: GPU with at least 8 GB memory is recommended for efficient execution.
- **Remote Access**: Accessible on cloud platforms (AWS, Google Cloud, etc.) if a physical GPU is unavailable.
- **Memory Requirements**: ~16 GB system RAM and ~20 GB of free disk space.

### Software Requirements
- **Operating System**: Linux/Ubuntu (preferred), macOS, or Windows.
- **Dependencies**:
  - Python >= 3.8
  - PyTorch >= 1.10
  - Opacus
  - Scikit-learn
  - NumPy, Matplotlib, TQDM
- Installation of dependencies can be automated using:
  ```bash
  pip install -r requirements.txt
  ```

### Estimated Time and Storage Consumption
- **Training Time**: Highly dependent on model type, dataset, and computational resources:
GPU: Training time is significantly faster, with larger networks feasible.
CPU: Larger networks might be infeasible, with extended training times.
DPSGD Impact: Significantly increases training time compared to non-private methods.
- **Storage**: ~8 GB for logs, intermediate checkpoints, and results.

## Environment

### Accessibility
The artifacts are available in the GitHub repository under the name **Neuromorphic_Privacy**:
- Repository: [GitHub Link](https://github.com/AyanaMoshruba/Neuromorphic_Privacy)
- Specific Commit: `abc12345` (replace with actual commit hash)

### Set up the environment
Clone the repository and install the dependencies:
```bash
git clone https://github.com/AyanaMoshruba/Neuromorphic_Privacy.git
cd Neuromorphic_Privacy
pip install -r requirements.txt
```
Ensure that the Python environment is activated and all dependencies are installed successfully.

### Testing the Environment
Run the following command to test the environment:
```bash
python tests/test_environment.py
```
Expected Output:
- "Environment setup successful."
- No errors during execution.

## Artifact Evaluation

### Main Results and Claims
#### Main Result 1: Privacy Resilience of SNNs and ANNs
The artifact demonstrates the effectiveness of DPSGD in reducing Membership Inference Attack (MIA) success rates across both SNN and ANN architectures. Refer to Section 5 in the paper for details.

#### Main Result 2: Impact of DPSGD on Privacy
DPSGD significantly reduces MIA success rates for both SNNs and ANNs, validating its efficacy as a privacy-preserving technique but also reduces test accuracy. Refer to Section 5.4 in the paper.

### Experiments
#### Experiment 1: Membership Inference Attack Evaluation (MIA- ANN) (Baseline)
- **Steps**:
  ```bash
  cd MIA/Baseline/ANN
  python train.py --dataset cifar10 --num_epochs 20 --batch_size 128 --data_path ./data --output_dir ./output
  ```
- **Expected Results**: AUC and ROC curve plots for attack success rates.
- **Runtime**: ~1 hour
- **Disk Space**: ~1 GB

#### Experiment 2: Membership Inference Attack Evaluation (MIA- SNN) (Baseline)
- **Steps**:
  ```bash
  cd MIA/Baseline/SNN
  python train.py --dataset cifar10 --num_epochs 20 --batch_size 128 --data_path ./data --output_dir ./output
  ```
#### Experiment 3: Membership Inference Attack Evaluation (MIA- ANN) (Deeper_Networks)
- **Steps**:
  ```bash
  cd MIA/Deeper_Networks/ANN
  python train_mia.py --dataset cifar10 --num_epochs 20 --model resnet18
  ```
- **Expected Results**: AUC and ROC curve plots for attack success rates.
- **Runtime**: ~7 hour
- **Disk Space**: ~2 GB

#### Experiment 4: Membership Inference Attack Evaluation (MIA- SNN) (Deeper_Networks)
- **Steps**:
  ```bash
  cd MIA/Deeper_Networks/SNN
  python train_mia.py --dataset cifar10 --arch gsresnet --num_steps 50 --num_epochs 70 --batch_size 128 --lr 0.0268
  ```
- **Expected Results**: AUC and ROC curve plots for attack success rates.
- **Runtime**: ~3 days 5 hours
- **Disk Space**: ~2GB

#### Experiment 5: DPSGD Training on CIFAR10 (ANN)
- **Steps**:
  ```bash
  cd DPSGD/ANN
  python train.py --dataset cifar10 --num_epochs 50 --epsilon 2.0
  ```
- **Expected Results**: Logs showing training accuracy, test accuracy, and privacy budget (epsilon).
- **Runtime**: ~2 hours
- **Disk Space**: ~2 GB

#### Experiment 6: DPSGD Training on CIFAR10 (SNN)
- **Steps**:
  ```bash
  cd DPSGD/SNN
  python train.py --dataset cifar10 --num_epochs 50 --epsilon 2.0
  ```
- **Expected Results**: Logs showing training accuracy, test accuracy, and privacy budget (epsilon).
- **Runtime**: ~5 hours
- **Disk Space**: ~2 GB


## Limitations
- The artifact includes results for selected datasets. Extensions to other datasets might require additional preprocessing steps.
- The TENNLab framework is not publicly available, so evolutionary algorithm codes could not be included.

## Notes on Reusability
The artifact is modular and can be reused for:
- Applying DPSGD to other neural architectures.
- Extending privacy evaluations to additional datasets.
- Incorporating differentially private training techniques for spiking neural networks.
Replace datasets in `utils.py` and modify `train.py` scripts for further extensions.
