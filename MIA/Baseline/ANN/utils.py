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

# Separate scaler for shadow datasets
shadow_scaler = StandardScaler()

def get_shadow_trainloader(dataset, subset_fraction=0.8, batch_size=128):
    global shadow_scaler

    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    elif dataset == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    elif dataset == "mnist":
        transform_train = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])
        full_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)

    elif dataset == "fmnist":
        transform_train = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])
        full_trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)

    elif dataset == "iris":
        # Load and preprocess the iris dataset
        iris_data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=42)

        # Fit scaler on shadow training data and transform
        X_train = shadow_scaler.fit_transform(X_train)

        # Convert the datasets to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)

        # Create DataLoader for the shadow training set
        full_trainset = TensorDataset(X_train, y_train)

    elif dataset == "breast_cancer":
        # Load and preprocess the breast cancer dataset
        cancer_data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.2, random_state=42)

        # Fit scaler on shadow training data and transform
        X_train = shadow_scaler.fit_transform(X_train)

        # Convert the datasets to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)

        # Create DataLoader for the shadow training set
        full_trainset = TensorDataset(X_train, y_train)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Generate random indices for shadow model subset
    num_train = len(full_trainset)
    indices = list(range(num_train))
    random.shuffle(indices)

    split = int(subset_fraction * num_train)
    subset_indices = indices[:split]

    # Using Subset class to create a dataset based on random indices
    train_subset = Subset(full_trainset, subset_indices)

    # Creating dataloader from subset
    shadow_trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    return shadow_trainloader


def get_shadow_testloader(dataset, subset_fraction=0.2, batch_size=100):
    global shadow_scaler

    if dataset == "cifar10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        full_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
   
    elif dataset == "cifar100":
        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        full_testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    elif dataset == "mnist":
        transform_test = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])
        full_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    elif dataset == "fmnist":
        transform_test = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])
        full_testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)

    elif dataset == "iris":
        # Load and preprocess the iris dataset
        iris_data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=42)

        # Transform shadow test set using the fitted shadow scaler
        X_test = shadow_scaler.transform(X_test)

        # Convert the datasets to PyTorch tensors
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # Create DataLoader for the shadow testing set
        full_testset = TensorDataset(X_test, y_test)

    elif dataset == "breast_cancer":
        # Load and preprocess the breast cancer dataset
        cancer_data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.2, random_state=42)

        # Transform shadow test set using the fitted shadow scaler
        X_test = shadow_scaler.transform(X_test)

        # Convert the datasets to PyTorch tensors
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # Create DataLoader for the shadow testing set
        full_testset = TensorDataset(X_test, y_test)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Generate random indices for shadow model subset
    num_test = len(full_testset)
    indices = list(range(num_test))
    random.shuffle(indices)

    split = int(subset_fraction * num_test)
    subset_indices = indices[:split]

    # Using Subset class to create a dataset based on random indices
    test_subset = Subset(full_testset, subset_indices)

    # Creating dataloader from subset
    shadow_testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return shadow_testloader