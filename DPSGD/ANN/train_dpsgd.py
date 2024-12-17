import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import get_trainloader, get_testloader
import argparse

# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Argument parser for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="ANN Model Training with DPSGD")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'mnist', 'fmnist', 'iris', 'breast_cancer'],
                        help='Dataset to use: cifar10, cifar100, mnist, fmnist, iris, breast_cancer')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epsilon', type=float, default=2.0, help='Privacy budget for DPSGD')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model and results')
    return parser.parse_args()


# Fully Connected Network for tabular datasets
class ANNFCNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(ANNFCNet, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x


# Convolutional Network for image datasets
class ANNConvNet(nn.Module):
    def __init__(self, dataset):
        super(ANNConvNet, self).__init__()

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
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x


# Function to choose the appropriate model based on the dataset
def choose_model(dataset):
    if dataset in ['cifar10', 'cifar100', 'mnist', 'fmnist']:
        return ANNConvNet(dataset)
    elif dataset == 'iris':
        return ANNFCNet(num_inputs=4, num_hidden=1000, num_outputs=3)
    elif dataset == 'breast_cancer':
        return ANNFCNet(num_inputs=30, num_hidden=1000, num_outputs=2)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# Training script
def train_dpsgd():
    args = parse_args()

    # Load data
    train_loader = get_trainloader(args.dataset, batch_size=args.batch_size)
    test_loader = get_testloader(args.dataset, batch_size=args.batch_size)

    # Initialize model, loss, and optimizer
    model = choose_model(args.dataset).to(device)
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
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1)
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
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        test_accuracy = 100.0 * correct / total
        print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    train_dpsgd()
