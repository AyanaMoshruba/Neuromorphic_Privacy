import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
import pickle
import time
from utils import get_trainloader, get_testloader
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import itertools
import argparse
from torch.optim import Adam
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



# Argument parser for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="SNN Model Training with DPSGD")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'mnist', 'fmnist', 'iris', 'breast_cancer'],
                        help='Dataset to use: cifar10, cifar100, mnist, fmnist, iris, breast_cancer')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of timesteps for spiking dynamics')
    parser.add_argument('--beta', type=float, default=0.95, help='Decay rate for LIF neurons')
    parser.add_argument('--epsilon', type=float, default=2.0, help='Privacy budget for DPSGD')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')  # Add this line
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model and results')
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
def choose_model(dataset, num_steps=10, beta=0.95):
    if dataset in ['cifar10', 'cifar100', 'mnist', 'fmnist']:
        return SNNConvNet(dataset=dataset, beta=beta, num_steps=num_steps)
    elif dataset == 'iris':
        return SNNFCNet(num_inputs=4, num_hidden=1000, num_outputs=3, beta=beta, num_steps=num_steps)
    elif dataset == 'breast_cancer':
        return SNNFCNet(num_inputs=30, num_hidden=1000, num_outputs=2, beta=beta, num_steps=num_steps)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# Training script
def train_dpsgd():
    args = parse_args()

    # Load data
    train_loader = get_trainloader(args.dataset, batch_size=args.batch_size)
    test_loader = get_testloader(args.dataset, batch_size=args.batch_size)

    # Initialize model, loss, and optimizer
    model = choose_model(args.dataset, num_steps=args.num_steps, beta=args.beta).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Configure PrivacyEngine
    privacy_engine = PrivacyEngine(secure_mode=False)
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=args.num_epochs,
        target_epsilon=args.epsilon,
        target_delta=1e-5,
        max_grad_norm=1.0,
    )

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            spk_rec, mem_rec = model(data)
            loss = criterion(mem_rec[-1], target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = mem_rec[-1].argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        train_accuracy = 100.0 * correct / total
        epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {total_loss:.4f}, Accuracy: {train_accuracy:.2f}%, ε: {epsilon:.2f}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                spk_rec, mem_rec = model(data)
                pred = mem_rec[-1].argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        test_accuracy = 100.0 * correct / total
        print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    train_dpsgd()