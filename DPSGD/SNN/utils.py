import torchvision
import torchvision.transforms as transforms
import torch
import random
from torch.utils.data import Subset, DataLoader, TensorDataset
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Global variable for scaler to ensure consistency
scaler = StandardScaler()

def get_trainloader(dataset, batch_size=128):
    global scaler

    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    elif dataset == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    elif dataset == "mnist":
        transform_train = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)

    elif dataset == "fmnist":
        transform_train = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)

    elif dataset == "iris":
        iris_data = load_iris()
        X_train, _, y_train, _ = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=42)
        X_train = scaler.fit_transform(X_train)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        trainset = TensorDataset(X_train, y_train)

    elif dataset == "breast_cancer":
        cancer_data = load_breast_cancer()
        X_train, _, y_train, _ = train_test_split(cancer_data.data, cancer_data.target, test_size=0.2, random_state=42)
        X_train = scaler.fit_transform(X_train)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        trainset = TensorDataset(X_train, y_train)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    return trainloader

def get_testloader(dataset, batch_size=100):
    global scaler

    if dataset == "cifar10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    elif dataset == "cifar100":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    elif dataset == "mnist":
        transform_test = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    elif dataset == "fmnist":
        transform_test = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)

    elif dataset == "iris":
        iris_data = load_iris()
        _, X_test, _, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=42)
        X_test = scaler.transform(X_test)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        testset = TensorDataset(X_test, y_test)

    elif dataset == "breast_cancer":
        cancer_data = load_breast_cancer()
        _, X_test, _, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.2, random_state=42)
        X_test = scaler.transform(X_test)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        testset = TensorDataset(X_test, y_test)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return testloader