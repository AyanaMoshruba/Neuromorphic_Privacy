if __name__ == '__main__':
    import torch.optim as optim
    import torchvision
    from torch.utils.data.dataloader import DataLoader
    from model import *
    from utils import *
    import argparse
    import os.path
    import numpy as np
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pickle
    import time
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    import multiprocessing as mp
    import logging
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import os
    import pandas as pd

    if __name__ == '__main__':
        mp.set_start_method('fork', force=True)

    torch.backends.cudnn.benchmark = True


    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser(description="Train ResNet6 on CIFAR10 or CIFAR100")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100","mnist", "fmnist"], required=True, help="Dataset to use: CIFAR10 or CIFAR100")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")   
    parser.add_argument("--model", choices=["resnet6", "resnet18", "resnet50", "vgg16"], required=True, help="Model to use: ResNet6 or ResNet18")

    args = parser.parse_args()


    # Load data
    trainloader = get_trainloader(args.dataset)
    testloader = get_testloader(args.dataset)

    shadow_trainloader = get_shadow_trainloader(args.dataset, subset_fraction=0.8)  # Change subset_fraction to whatever you need

    # Number of classes
    if args.dataset == "cifar10":
        num_classes = 10
        num_channels = 3
    elif args.dataset == "cifar100":
        num_classes = 100
        num_channels = 3
    elif args.dataset in ["mnist", "fmnist"]:
        num_classes = 10
        num_channels = 1
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")      

    # Choose model
    if args.model == "resnet6":
        model = ResNet6(num_classes=num_classes, num_channels=num_channels).to(device)
    elif args.model == "resnet18":
        model = ResNet18(num_classes=num_classes, num_channels=num_channels).to(device)
    elif args.model == "resnet50":
        model = ResNet50(num_classes=num_classes, num_channels=num_channels).to(device)
    elif args.model == "vgg16":
        model = VGG16(num_classes=num_classes, num_channels=num_channels).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    

    target_model = model.to(device)
    shadow_model = model.to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    target_optimizer = optim.SGD(target_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    shadow_optimizer = optim.SGD(shadow_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    loss = nn.CrossEntropyLoss()

    # Define training function
    def train(model, trainloader, optimizer, epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        losses = []
        
        for i, data in enumerate(tqdm(trainloader, 0)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate and print train accuracy and loss
        accuracy = 100 * correct / total
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} \t Train Accuracy: {accuracy:.2f}%")


    # Define testing function
    def test(model, testloader):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("\nTest set:  Accuracy: {}/{} ({:.2f}%)\n".format(
                correct,
                total,
                100.0 * correct / total,
            ))

        accuracy = 100 * correct / total
        return accuracy

    # Train and test the target_model
    avr_acc_target_model = []
    for epoch in range(1, args.num_epochs + 1):
        train(target_model, trainloader, target_optimizer, epoch)
        acc = test(target_model, testloader)
        avr_acc_target_model.append(acc)

    print("Average accuracy of target model: {}".format(np.mean(avr_acc_target_model)))

    # Train and test the shadow_model
    avr_acc_shadow_model = []
    for epoch in range(1, args.num_epochs + 1):
        train(shadow_model, shadow_trainloader, shadow_optimizer, epoch)
        acc = test(shadow_model, testloader)
        avr_acc_shadow_model.append(acc)

    print("Average accuracy of shadow model: {}".format(np.mean(avr_acc_shadow_model)))



    import numpy as np
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    # Assuming the shadow_model is already trained and available
    # Assuming shadow_train_loader and shadow_test_loader are defined and provide data in the CIFAR-10 format

    # num_classes = 10
    x_data_shadow, y_data_shadow = [[] for _ in range(num_classes)], [[] for _ in range(num_classes)]

    shadow_model.eval()  # Set the shadow model to evaluation mode
    with torch.no_grad():
        # Query the shadow model with its training dataset
        for dataX, datay in shadow_trainloader:
            dataX = dataX.to(device)
            datay = datay.to(device)

            # Get the prediction vectors from the shadow model
            outputs = shadow_model(dataX)  # Adjust if your model's output structure is different
            # Assuming outputs are logits, convert to probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Add labeled records for the shadow model's training dataset
            for i in range(dataX.shape[0]):
                idx = datay[i].item()  # get the class index
                y_data_shadow[idx].append(1)  # '1' for in training set
                x_data_shadow[idx].append(probabilities[i].cpu().numpy())  # probabilities are used for MIA

        # Query the shadow model with a disjoint test set
        for dataX, datay in testloader:
            dataX = dataX.to(device)
            datay = datay.to(device)

            # Get the prediction vectors from the shadow model
            outputs = shadow_model(dataX)  # Adjust if your model's output structure is different
            # Assuming outputs are logits, convert to probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Add labeled records for the shadow model's testing dataset
            for i in range(dataX.shape[0]):
                idx = datay[i].item()  # get the class index
                y_data_shadow[idx].append(0)  # '0' for in testing set
                x_data_shadow[idx].append(probabilities[i].cpu().numpy())  # probabilities are used for MIA
    # New lists for balanced data
    x_data_balanced = [[] for _ in range(num_classes)]
    y_data_balanced = [[] for _ in range(num_classes)]

    for idx in range(num_classes):
        # Combine x and y data for only '1' labels
        combined_data_1 = [(x, y) for x, y in zip(x_data_shadow[idx], y_data_shadow[idx]) if y == 1]
        # Get all data for '0' labels
        combined_data_0 = [(x, y) for x, y in zip(x_data_shadow[idx], y_data_shadow[idx]) if y == 0]

        # Downsample '1' labels using random.sample
        downsampled_data_1 = random.sample(combined_data_1, len(combined_data_0))

        # Combine downsampled '1' labels with '0' labels
        combined_balanced = downsampled_data_1 + combined_data_0

        # Split and append the balanced data back to x_data_balanced and y_data_balanced
        for item in combined_balanced:
            x_data_balanced[idx].append(item[0])
            y_data_balanced[idx].append(item[1])


    # Convert lists to numpy arrays or perform further processing as needed
    x_data_shadow = [np.array(x_data) for x_data in x_data_balanced]
    y_data_shadow = [np.array(y_data) for y_data in y_data_balanced]      


    x_data_target, y_data_target = [[] for _ in range(num_classes)], [[] for _ in range(num_classes)]

    target_model.eval()  # Set the shadow model to evaluation mode
    with torch.no_grad():
        # Query the shadow model with its training dataset
        for dataX, datay in trainloader:
            dataX = dataX.to(device)
            datay = datay.to(device)

            # Get the prediction vectors from the shadow model
            outputs = target_model(dataX)  # Adjust if your model's output structure is different
            # Assuming outputs are logits, convert to probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Add labeled records for the shadow model's training dataset
            for i in range(dataX.shape[0]):
                idx = datay[i].item()  # get the class index
                y_data_target[idx].append(1)  # '1' for in training set
                x_data_target[idx].append(probabilities[i].cpu().numpy())  # probabilities are used for MIA

        # Query the shadow model with a disjoint test set
        for dataX, datay in testloader:
            dataX = dataX.to(device)
            datay = datay.to(device)

            # Get the prediction vectors from the shadow model
            outputs = target_model(dataX)  # Adjust if your model's output structure is different
            # Assuming outputs are logits, convert to probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Add labeled records for the shadow model's testing dataset
            for i in range(dataX.shape[0]):
                idx = datay[i].item()  # get the class index
                y_data_target[idx].append(0)  # '0' for in testing set
                x_data_target[idx].append(probabilities[i].cpu().numpy())  # probabilities are used for MIA

        
    # New lists for balanced data
    x_data_balanced = [[] for _ in range(num_classes)]
    y_data_balanced = [[] for _ in range(num_classes)]

    for idx in range(num_classes):
        # Combine x and y data for only '1' labels
        combined_data_1 = [(x, y) for x, y in zip(x_data_target[idx], y_data_target[idx]) if y == 1]
        # Get all data for '0' labels
        combined_data_0 = [(x, y) for x, y in zip(x_data_target[idx], y_data_target[idx]) if y == 0]

        # Downsample '1' labels using random.sample
        downsampled_data_1 = random.sample(combined_data_1, len(combined_data_0))

        # Combine downsampled '1' labels with '0' labels
        combined_balanced = downsampled_data_1 + combined_data_0

        # Split and append the balanced data back to x_data_balanced and y_data_balanced
        for item in combined_balanced:
            x_data_balanced[idx].append(item[0])
            y_data_balanced[idx].append(item[1])

    # Convert lists to numpy arrays or perform further processing as needed
    x_data_target = [np.array(x_data) for x_data in x_data_balanced]
    y_data_target = [np.array(y_data) for y_data in y_data_balanced]  


    from sklearn.svm import SVC
    # Build and train the SVM attack model
    attack_models= []
    score_list = []
    for i in range(10):
        attack_model = SVC(kernel='rbf')
        attack_model.fit(x_data_shadow[i], y_data_shadow[i])
        score = attack_model.score(x_data_target[i], y_data_target[i])
        print(f"score:{score}")
        attack_models.append(attack_model)

    # Define the base directory for saving ROC data
    roc_data_base_dir = "roc_data1"

    # Create a directory specific to the dataset
    dataset_dir = os.path.join(roc_data_base_dir, args.dataset)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for i in range(10):
        # Calculate decision scores or probabilities
        decision_scores = attack_models[i].decision_function(x_data_target[i])
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_data_target[i], decision_scores)
        roc_auc = auc(fpr, tpr)
        
        # Save ROC curve data to a CSV file
        roc_data_df = pd.DataFrame({
            'False Positive Rate': fpr,
            'True Positive Rate': tpr,
            'Thresholds': thresholds
        })
        roc_data_file = os.path.join(dataset_dir, f'class_{i}_roc_data.csv')
        roc_data_df.to_csv(roc_data_file, index=False)
        
        # Plotting the ROC curve
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Attack Model - Class {i}')
        plt.legend(loc="lower right")
        
        # Save the plot as an image file
        plot_file = os.path.join(dataset_dir, f'class_{i}_roc_curve.png')
        plt.savefig(plot_file)
        plt.close()

        print(f'Class {i} AUC: {roc_auc:.4f}')    