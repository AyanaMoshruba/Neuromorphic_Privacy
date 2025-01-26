import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from tqdm import tqdm
from utils import get_trainloader, get_testloader, get_shadow_trainloader, get_shadow_testloader
from sklearn.svm import SVC
from sklearn.metrics import precision_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import csv
import snntorch as snn
from snntorch import spikegen

# Argument parser for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="SNN Model Training with MIA")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'mnist', 'fmnist', 'iris', 'breast_cancer'],
                        help='Dataset to use: cifar10, cifar100, mnist, fmnist, iris, breast_cancer')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model and results')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of timesteps for spiking dynamics')
    parser.add_argument('--beta', type=float, default=0.95, help='Decay rate for LIF neurons')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to download the datasets')
    return parser.parse_args()


# Spiking Convolutional Network for image datasets
class SNNConvNet(nn.Module):
    def __init__(self, dataset, beta=0.95, num_steps=10):
        super(SNNConvNet, self).__init__()
        self.num_steps = num_steps
        self.beta = beta

        if dataset in ['cifar10', 'cifar100']:
            input_channels = 3
        else:
            input_channels = 1

        if dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'cifar10':
            num_classes = 10
        else:
            num_classes = 10

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        # Fully connected layers
        if dataset in ['cifar10', 'cifar100']:
            num_features = 64 * 8 * 8
        else:
            num_features = 64 * 7 * 7

        self.fc1 = nn.Linear(num_features, 1000)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(1000, num_classes)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Spiking layers
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# Spiking Fully Connected Network for tabular datasets
class SNNFCNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta=0.95, num_steps=10):
        super(SNNFCNet, self).__init__()
        self.num_steps = num_steps
        self.beta = beta

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


# Function to choose the appropriate model based on the dataset
def choose_model(dataset):
    if dataset in ['cifar10', 'cifar100', 'mnist', 'fmnist']:
        return SNNConvNet(dataset)
    elif dataset == 'iris':
        return SNNFCNet(num_inputs=4, num_hidden=1000, num_outputs=3)
    elif dataset == 'breast_cancer':
        return SNNFCNet(num_inputs=30, num_hidden=1000, num_outputs=2)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

# Set up output directories to save model and results
def setup_output_directories(dataset, output_dir, run_idx):
    base_dir = os.path.join(output_dir, dataset, f'run_{run_idx}')
    roc_data_dir = os.path.join(base_dir, 'roc_data')
    os.makedirs(roc_data_dir, exist_ok=True)
    return base_dir, roc_data_dir


# Training and evaluation logic for SNNs
def train_and_evaluate_snn(model, optimizer, loss_fn, train_loader, test_loader, device, num_epochs, model_name="SNN"):
    loss_hist, test_loss_hist, train_acc_hist, test_acc_hist = [], [], [], []
    best_test_accuracy = 0.0

    print(f"Starting training for {model_name}...")

    for epoch in range(num_epochs):
        model.train()
        total_train_correct, total_train_samples = 0, 0
        for data, targets in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            spk_rec, mem_rec = model(data)
            loss_val = loss_fn(mem_rec[-1], targets)
            loss_hist.append(loss_val.item())
            loss_val.backward()
            optimizer.step()
            _, predicted = torch.max(mem_rec[-1].data, 1)
            total_train_correct += (predicted == targets).sum().item()
            total_train_samples += targets.size(0)
        train_accuracy = (total_train_correct / total_train_samples) * 100
        train_acc_hist.append(train_accuracy)

        model.eval()
        total_test_correct, total_test_samples = 0, 0
        with torch.no_grad():
            for test_data, test_targets in tqdm(test_loader, desc=f"Testing Epoch {epoch + 1}/{num_epochs}"):
                test_data, test_targets = test_data.to(device), test_targets.to(device)
                test_spk, test_mem = model(test_data)
                test_loss = loss_fn(test_mem[-1], test_targets)
                test_loss_hist.append(test_loss.item())
                _, test_predicted = torch.max(test_mem[-1].data, 1)
                total_test_correct += (test_predicted == test_targets).sum().item()
                total_test_samples += test_targets.size(0)
        test_accuracy = (total_test_correct / total_test_samples) * 100
        test_acc_hist.append(test_accuracy)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy

        print(f"Epoch {epoch + 1}: {model_name} - Train Loss = {np.mean(loss_hist):.4f}, "
              f"Train Accuracy = {train_accuracy:.2f}%, Test Accuracy = {test_accuracy:.2f}%")
    
    print(f"Best Test Accuracy for {model_name}: {best_test_accuracy:.2f}%")
    return train_acc_hist, test_acc_hist, loss_hist, test_loss_hist


# Membership Inference Attack
def mia_attack(target_model, shadow_model, train_loader, test_loader, shadow_train_loader, shadow_test_loader, device, roc_data_dir, dataset):
    num_classes = 100 if dataset == 'cifar100' else 10

    x_data_shadow, y_data_shadow = [[] for _ in range(num_classes)], [[] for _ in range(num_classes)]
    x_data_target, y_data_target = [[] for _ in range(num_classes)], [[] for _ in range(num_classes)]

    # Collect shadow model outputs
    shadow_model.eval()
    with torch.no_grad():
        for loader, label in [(shadow_train_loader, 1), (shadow_test_loader, 0)]:
            for data, targets in loader:
                data, targets = data.to(device), targets.to(device)
                _, mem_rec = shadow_model(data)
                for i in range(data.size(0)):
                    class_idx = targets[i].item()
                    x_data_shadow[class_idx].append(mem_rec[-1][i].cpu().numpy())
                    y_data_shadow[class_idx].append(label)

    # Collect target model outputs
    target_model.eval()
    with torch.no_grad():
        for loader, label in [(train_loader, 1), (test_loader, 0)]:
            for data, targets in loader:
                data, targets = data.to(device), targets.to(device)
                _, mem_rec = target_model(data)
                for i in range(data.size(0)):
                    class_idx = targets[i].item()
                    x_data_target[class_idx].append(mem_rec[-1][i].cpu().numpy())
                    y_data_target[class_idx].append(label)

    # Initialize lists to store results
    auc_scores = []

    # Process each class separately
    for class_idx in range(num_classes):
        if len(x_data_shadow[class_idx]) == 0 or len(x_data_target[class_idx]) == 0:
            print(f"No data for class {class_idx}. Skipping...")
            continue

        # Flatten the data for each class
        x_shadow = np.array(x_data_shadow[class_idx])
        y_shadow = np.array(y_data_shadow[class_idx])
        x_target = np.array(x_data_target[class_idx])
        y_target = np.array(y_data_target[class_idx])

        # Train SVM attack model
        attack_model = SVC(kernel='rbf', probability=True)
        attack_model.fit(x_shadow, y_shadow)

        # Get decision scores and calculate ROC
        decision_scores = attack_model.decision_function(x_target)
        fpr, tpr, _ = roc_curve(y_target, decision_scores)
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)

        # Save ROC data and plot
        save_roc_data_per_class(fpr, tpr, roc_data_dir, class_idx, roc_auc)

        print(f"Class {class_idx} - AUC: {roc_auc:.4f}")

    print("Membership Inference Attack completed.")
    return auc_scores


# Save and plot ROC data for each class
def save_roc_data_per_class(fpr, tpr, roc_folder, class_idx, roc_auc):
    # Save ROC curve as CSV
    csv_path = os.path.join(roc_folder, f'roc_class_{class_idx}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['FPR', 'TPR'])
        writer.writerows(zip(fpr, tpr))

    # Save ROC plot
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class {class_idx}')
    plt.legend()
    plot_path = os.path.join(roc_folder, f'roc_curve_class_{class_idx}.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"ROC data and plot saved for class {class_idx}: AUC = {roc_auc:.4f}")
# Main function
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_trainloader(args.dataset, batch_size=args.batch_size)
    test_loader = get_testloader(args.dataset, batch_size=args.batch_size)
    shadow_train_loader = get_shadow_trainloader(args.dataset, batch_size=args.batch_size)
    shadow_test_loader = get_shadow_testloader(args.dataset, batch_size=args.batch_size)

    target_model = choose_model(args.dataset).to(device)
    shadow_model = choose_model(args.dataset).to(device)

    target_optimizer = optim.Adam(target_model.parameters(), lr=0.001)
    shadow_optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_and_evaluate_snn(target_model, target_optimizer, loss_fn, train_loader, test_loader, device, args.num_epochs, "Target Model")
    train_and_evaluate_snn(shadow_model, shadow_optimizer, loss_fn, shadow_train_loader, shadow_test_loader, device, args.num_epochs, "Shadow Model")

    roc_data_dir = os.path.join(args.output_dir, 'roc_data')
    os.makedirs(roc_data_dir, exist_ok=True)

    mia_attack(target_model, shadow_model, train_loader, test_loader, shadow_train_loader, shadow_test_loader, device, roc_data_dir, args.dataset)

if __name__ == "__main__":
    main()
