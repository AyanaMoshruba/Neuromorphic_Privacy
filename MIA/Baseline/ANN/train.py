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

# Argument parser for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="CNN Model Training")
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['cifar10', 'cifar100', 'mnist', 'fmnist', 'iris', 'breast_cancer'],
                       help='Dataset to use: cifar10, cifar100, mnist, fmnist, iris, breast_cancer')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to download the datasets')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model and results')
    return parser.parse_args()

# CNN for CIFAR10, CIFAR100, MNIST, and FMNIST
class ConvNet(nn.Module):
    def __init__(self, dataset):
        super(ConvNet, self).__init__()
        
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
            
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        if dataset in ['cifar10', 'cifar100']:
            num_features = 64 * 8 * 8
        else:
            num_features = 64 * 7 * 7
            
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Fully Connected Network for Iris and Breast Cancer
class FCNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to choose the appropriate model based on the dataset
def choose_model(dataset):
    if dataset in ['cifar10', 'cifar100', 'mnist', 'fmnist']:
        return ConvNet(dataset)
    elif dataset == 'iris':
        return FCNet(num_inputs=4, num_hidden=1000, num_outputs=3)
    elif dataset == 'breast_cancer':
        return FCNet(num_inputs=30, num_hidden=1000, num_outputs=2)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

# Set up output directories to save model and results
def setup_output_directories(dataset, output_dir, run_idx):
    base_dir = os.path.join(output_dir, dataset, f'run_{run_idx}')
    roc_data_dir = os.path.join(base_dir, 'roc_data')
    os.makedirs(roc_data_dir, exist_ok=True)
    return base_dir, roc_data_dir

# Training and evaluation logic remains the same
def train_and_evaluate_model(model, optimizer, loss_fn, train_loader, test_loader, device, num_epochs, model_name="Model"):
    loss_hist, test_loss_hist, train_acc_hist, test_acc_hist = [], [], [], []
    best_test_accuracy = 0.0
    
    print(f"Starting training for {model_name}...")

    for epoch in range(num_epochs):
        model.train()
        total_train_correct, total_train_samples = 0, 0
        for data, targets in tqdm(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss_val = loss_fn(output, targets)
            loss_hist.append(loss_val.item())
            loss_val.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, 1)
            total_train_correct += (predicted == targets).sum().item()
            total_train_samples += targets.size(0)
        train_accuracy = (total_train_correct / total_train_samples) * 100
        train_acc_hist.append(train_accuracy)

        model.eval()
        total_test_correct, total_test_samples = 0, 0
        with torch.no_grad():
            for test_data, test_targets in test_loader:
                test_data, test_targets = test_data.to(device), test_targets.to(device)
                output = model(test_data)
                test_loss = loss_fn(output, test_targets)
                test_loss_hist.append(test_loss.item())
                _, test_predicted = torch.max(output.data, 1)
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

# Function to save ROC data as CSV and plot
def save_roc_data(y_true, y_scores, roc_folder, model_num):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Save data as CSV
    with open(os.path.join(roc_folder, f'roc_model_{model_num}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['FPR', 'TPR'])
        writer.writerows(zip(fpr, tpr))

    # Save ROC plot
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Attack Model {model_num}')
    plt.legend()
    plt.savefig(os.path.join(roc_folder, f'roc_curve_model_{model_num}.png'))
    plt.close()

# MIA attack function with dataset-specific num_classes
def mia_attack(target_model, shadow_model, train_loader, test_loader, shadow_train_loader, shadow_test_loader, device, roc_data_dir, dataset):
    # Set number of classes based on dataset
    if dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'iris':
        num_classes = 3
    elif dataset == 'breast_cancer':
        num_classes = 2
    else:
        num_classes = 10

    # Collect shadow model outputs for training data (label 1)
    shadow_model.eval()
    x_data_shadow, y_data_shadow = [[] for _ in range(num_classes)], [[] for _ in range(num_classes)]

    with torch.no_grad():
        for data, targets in shadow_train_loader:
            data = data.to(device)
            targets = targets.to(device)

            shadow_output= shadow_model(data)

            for i in range(data.size(0)):
                idx = targets[i].item()
                if idx < num_classes:  # Ensure idx is within the valid range
                    y_data_shadow[idx].append(1)  # Training data label
                    x_data_shadow[idx].append(shadow_output[i].flatten().cpu().numpy())

    # Collect shadow model outputs for testing data (label 0)
    with torch.no_grad():
        for data, targets in shadow_test_loader:
            data = data.to(device)
            targets = targets.to(device)

            shadow_output = shadow_model(data)

            for i in range(data.size(0)):
                idx = targets[i].item()
                if idx < num_classes:  # Ensure idx is within the valid range
                    y_data_shadow[idx].append(0)  # Testing data label
                    x_data_shadow[idx].append(shadow_output[i].flatten().cpu().numpy())

    # Balance the shadow attack dataset
    x_data_balanced_shadow, y_data_balanced_shadow = [[] for _ in range(num_classes)], [[] for _ in range(num_classes)]

    for idx in range(num_classes):
        combined_data_1 = [(x, y) for x, y in zip(x_data_shadow[idx], y_data_shadow[idx]) if y == 1]
        combined_data_0 = [(x, y) for x, y in zip(x_data_shadow[idx], y_data_shadow[idx]) if y == 0]

        # Downsample or balance the data
        if len(combined_data_1) < len(combined_data_0):
            downsampled_data_0 = random.sample(combined_data_0, len(combined_data_1))
            combined_balanced = combined_data_1 + downsampled_data_0
        else:
            downsampled_data_1 = random.sample(combined_data_1, len(combined_data_0))
            combined_balanced = downsampled_data_1 + combined_data_0

        for x, y in combined_balanced:
            x_data_balanced_shadow[idx].append(x)
            y_data_balanced_shadow[idx].append(y)

    # Prepare target attack dataset
    # Collect target model outputs for training data (label 1)
    target_model.eval()
    x_data_target, y_data_target = [[] for _ in range(num_classes)], [[] for _ in range(num_classes)]

    with torch.no_grad():
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)

            target_output= target_model(data)

            for i in range(data.size(0)):
                idx = targets[i].item()
                if idx < num_classes:  # Ensure idx is within the valid range
                    y_data_target[idx].append(1)  # Training data label
                    x_data_target[idx].append(target_output[i].flatten().cpu().numpy())

    # Collect target model outputs for testing data (label 0)
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            target_output = target_model(data)

            for i in range(data.size(0)):
                idx = targets[i].item()
                if idx < num_classes:  # Ensure idx is within the valid range
                    y_data_target[idx].append(0)  # Testing data label
                    x_data_target[idx].append(target_output[i].flatten().cpu().numpy())

    # Balance the target attack dataset
    x_data_balanced_target, y_data_balanced_target = [[] for _ in range(num_classes)], [[] for _ in range(num_classes)]

    for idx in range(num_classes):
        combined_data_1 = [(x, y) for x, y in zip(x_data_target[idx], y_data_target[idx]) if y == 1]
        combined_data_0 = [(x, y) for x, y in zip(x_data_target[idx], y_data_target[idx]) if y == 0]

        # Downsample or balance the data
        if len(combined_data_1) < len(combined_data_0):
            downsampled_data_0 = random.sample(combined_data_0, len(combined_data_1))
            combined_balanced = combined_data_1 + downsampled_data_0
        else:
            downsampled_data_1 = random.sample(combined_data_1, len(combined_data_0))
            combined_balanced = downsampled_data_1 + combined_data_0

        for x, y in combined_balanced:
            x_data_balanced_target[idx].append(x)
            y_data_balanced_target[idx].append(y)

    # Convert lists to numpy arrays
    x_data_shadow = [np.array(x_data_balanced_shadow[i]) for i in range(num_classes)]
    y_data_shadow = [np.array(y_data_balanced_shadow[i]) for i in range(num_classes)]
    x_data_target = [np.array(x_data_balanced_target[i]) for i in range(num_classes)]
    y_data_target = [np.array(y_data_balanced_target[i]) for i in range(num_classes)]

    # Build and train the SVM attack models
    attack_models = []
    score_list = []
    precision_list = []
    f1_score_list = []

    for i in range(num_classes):
        if len(x_data_shadow[i]) == 0 or len(x_data_target[i]) == 0:
            print(f"No data for class {i}. Skipping...")
            continue

        attack_model = SVC(kernel='rbf', probability=True)
        attack_model.fit(x_data_shadow[i], y_data_shadow[i])

        # Make predictions on the target data
        predictions = attack_model.predict(x_data_target[i])

        # Calculate the score (accuracy)
        score = attack_model.score(x_data_target[i], y_data_target[i])
        print(f"Class {i} - Score (Accuracy): {score}")

        # Calculate precision and F1 score
        precision = precision_score(y_data_target[i], predictions, zero_division=0)
        f1 = f1_score(y_data_target[i], predictions, zero_division=0)

        print(f"Class {i} - Precision: {precision}")
        print(f"Class {i} - F1 Score: {f1}")

        # Append the results to their respective lists
        attack_models.append(attack_model)
        score_list.append(score)
        precision_list.append(precision)
        f1_score_list.append(f1)

    # Save ROC curves and data
    for i in range(num_classes):
        if len(x_data_target[i]) == 0:
            print(f"No data for class {i} in target dataset. Skipping ROC curve...")
            continue

        # Calculate decision scores or probabilities
        decision_scores = attack_models[i].decision_function(x_data_target[i])

        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_data_target[i], decision_scores)
        roc_auc = auc(fpr, tpr)
        print(f"Class {i} AUC: {roc_auc:.4f}")
        # Save ROC data as CSV
        with open(os.path.join(roc_data_dir, f'roc_model_{i}.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['FPR', 'TPR'])
            writer.writerows(zip(fpr, tpr))

        # Save ROC plot
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Attack Model Class {i}\nAUC = {roc_auc:.2f}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(roc_data_dir, f'roc_curve_model_{i}.png'))
        plt.close()

    print("Membership Inference Attack completed and ROC data saved.")

# Main function
def main():
    args = parse_args()
    print("Arguments parsed successfully")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading datasets...")
    train_loader = get_trainloader(args.dataset, batch_size=args.batch_size)
    test_loader = get_testloader(args.dataset, batch_size=args.batch_size)
    shadow_train_loader = get_shadow_trainloader(args.dataset, batch_size=args.batch_size)
    shadow_test_loader = get_shadow_testloader(args.dataset, batch_size=args.batch_size)
    print("Datasets loaded")

    # Set up directories for a single run
    output_dir, roc_data_dir = setup_output_directories(args.dataset, args.output_dir, run_idx=1)

    # Initialize target and shadow models
    target_model = choose_model(args.dataset).to(device)
    shadow_model = choose_model(args.dataset).to(device)

    target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.0001)
    shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    # Train and evaluate models
    print("Training Target Model...")
    target_train_acc, target_test_acc, _, _ = train_and_evaluate_model(
        target_model, target_optimizer, loss_fn, train_loader, test_loader, 
        device, args.num_epochs, model_name="Target Model"
    )

    print("Training Shadow Model...")
    shadow_train_acc, shadow_test_acc, _, _ = train_and_evaluate_model(
        shadow_model, shadow_optimizer, loss_fn, shadow_train_loader, shadow_test_loader,
        device, args.num_epochs, model_name="Shadow Model"
    )

    # Perform MIA
    print("Performing Membership Inference Attack...")
    mia_attack(target_model, shadow_model, train_loader, test_loader, 
              shadow_train_loader, shadow_test_loader, device, roc_data_dir, args.dataset)
    print(f"MIA results saved in {roc_data_dir}")

if __name__ == "__main__":
    print("Starting main function")
    main()