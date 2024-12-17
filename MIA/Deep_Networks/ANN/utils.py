import torchvision
import torchvision.transforms as transforms
import torch
import random
from torch.utils.data import Subset

def get_trainloader(dataset, batch_size=128):
    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset ==  "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    elif dataset == "mnist":
        transform_train = transforms.Compose([
            transforms.Resize(32),  # Resizing MNIST images to 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalization specific to MNIST
        ])
    elif dataset == "fmnist":
        transform_train = transforms.Compose([
            transforms.Resize(32),  # Resizing FMNIST images to 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalization specific to Fashion-MNIST
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    elif dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    elif dataset == "mnist":
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    elif dataset == "fmnist":
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return trainloader

def get_testloader(dataset, batch_size=100):
    if dataset == "cifar10":
        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset == "cifar100":
        transform_test= transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    elif dataset == "mnist":
        transform_test = transforms.Compose([
            transforms.Resize(32),  # Resizing MNIST images to 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalization specific to MNIST
        ])
    elif dataset == "fmnist":
        transform_test = transforms.Compose([
            transforms.Resize(32),  # Resizing FMNIST images to 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalization specific to Fashion-MNIST
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if dataset == "cifar10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == "cifar100":
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == "mnist":
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == "fmnist":
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return testloader

def get_shadow_trainloader(dataset, subset_fraction=0.9, batch_size=128):
    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    elif dataset == "mnist":
        transform_train = transforms.Compose([
            transforms.Resize(32),  # Resizing MNIST images to 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalization specific to MNIST
        ])
    elif dataset == "fmnist":
        transform_train = transforms.Compose([
            transforms.Resize(32),  # Resizing FMNIST images to 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalization specific to Fashion-MNIST
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Load the full dataset
    if dataset == "cifar10":
        full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    elif dataset == "cifar100":
        full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    elif dataset == "mnist":
        full_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    elif dataset == "fmnist":
        full_trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Generate random indices
    num_train = len(full_trainset)
    indices = list(range(num_train))
    random.shuffle(indices)

    split = int(subset_fraction * num_train)
    subset_indices = indices[:split]

    # Using Subset class to create a dataset based on random indices
    train_subset = Subset(full_trainset, subset_indices)

    # Creating dataloader from subset
    shadow_trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return shadow_trainloader
