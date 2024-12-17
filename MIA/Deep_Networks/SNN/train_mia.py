
if __name__ == '__main__':
    import torch.optim as optim
    import torchvision
    from torch.utils.data.dataloader import DataLoader
    from torchvision import transforms
    from model import *
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    import argparse
    import os.path
    import numpy as np
    from utils import *
    import multiprocessing as mp
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import os
    import itertools
    from tqdm import tqdm
    import glob
    import math
    import cv2

    import torch
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader, Subset
    import random
    import pandas as pd
    from PIL import Image
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import utils, nn, optim
    import snntorch as snn
    from snntorch import surrogate
    import snntorch.functional as SF
    from torchvision import datasets, transforms

    if __name__ == '__main__':
        mp.set_start_method('fork', force=True)
    torch.backends.cudnn.benchmark = True

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   


    # Parse input arguments
    parser = argparse.ArgumentParser(description='Train S-ResNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_folder',           default='data', type=str, help='Folder for saving data')
    parser.add_argument('--weight_folder',           default='models', type=str, help='Folder for saving weights')
    parser.add_argument('--reload',                default=None, type=str, help='Path to weights to reload')
    parser.add_argument('--fine_tune',             default=False, action='store_true',
                        help='Does not reload conv1, FC and starts from epoch0')
    parser.add_argument('--seed',                  default=0,        type=int,   help='Random seed')
    parser.add_argument('--num_steps',             default=50,    type=int, help='Number of time-step')
    parser.add_argument('--batch_size',            default=21,       type=int,   help='Batch size')
    parser.add_argument('--lr',                    default=0.0268,   type=float, help='Learning rate')
    parser.add_argument('--leak_mem',              default=0.874,   type=float, help='Membrane leakage')
    parser.add_argument('--arch',                  default='sresnet',   type=str, help='[sresnet, sresnet_nm, gsresnet, spikingvgg16]')
    parser.add_argument('--n',                     default=6, type=int, help='Depth scaling of the S-ResNet')
    parser.add_argument('--nFilters',              default=32, type=int, help='Width scaling of the S-ResNet')
    parser.add_argument('--boosting',              default=False, action='store_true', help='Use boosting layer')
    parser.add_argument('--dataset',               default='cifar100',   type=str,
                        help='[cifar10, cifar100, mnist, fmnist, cifar10dvs]')
    parser.add_argument('--num_epochs',            default=70,       type=int,   help='Number of epochs')
    parser.add_argument('--num_workers',           default=1, type=int, help='Number of workers')
    parser.add_argument('--train_display_freq',    default=1, type=int, help='Display_freq for train')
    parser.add_argument('--test_display_freq',     default=1, type=int, help='Display_freq for test')
    # parser.add_argument('--device',                default='0', type=str, help='GPU to use')
    parser.add_argument('--poisson_gen',           default=False, action='store_true', help='Use poisson spike generation')

    global args
    args = parser.parse_args()

    # Define folder where weights are saved
    if os.path.isdir(args.weight_folder) is not True:
        os.mkdir(args.weight_folder)

    # Define folder where data is saved
    if os.path.isdir(args.data_folder) is not True:
        os.mkdir(args.data_folder)

    experiment_name = (args.dataset)+'_'+(args.arch)+'_timestep'+str(args.num_steps) +'_lr'+str(args.lr) + '_epoch' + str(args.num_epochs) + '_leak' + str(args.leak_mem)

    # Initialize random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Network parameters
    leak_mem = args.leak_mem
    batch_size      = args.batch_size
    batch_size_test = args.batch_size*2
    num_epochs      = args.num_epochs
    num_steps       = args.num_steps
    lr   = args.lr

    # Load  dataset

    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0, 0, 0), (1, 1, 1)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0, 0, 0), (1, 1, 1)),
    # ])
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        num_cls = 10
        img_size = 32

        train_set = torchvision.datasets.CIFAR10(root=args.data_folder, train=True,
                                                download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root=args.data_folder, train=False,
                                                download=True, transform=transform_test)
    elif args.dataset == 'cifar100':
        num_cls = 100
        img_size = 32

        train_set = torchvision.datasets.CIFAR100(root=args.data_folder, train=True,
                                                download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root=args.data_folder, train=False,
                                                download=True, transform=transform_test)
    elif args.dataset == 'mnist':
        num_cls = 10
        img_size = 32

        transform_mnist = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_set = torchvision.datasets.MNIST(root=args.data_folder, train=True,
                                            download=True, transform=transform_mnist)
        test_set = torchvision.datasets.MNIST(root=args.data_folder, train=False,
                                            download=True, transform=transform_mnist)
    elif args.dataset == 'fmnist':
        num_cls = 10
        img_size = 32

        transform_fmnist = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])

        train_set = torchvision.datasets.FashionMNIST(root=args.data_folder, train=True,
                                                    download=True, transform=transform_fmnist)
        test_set = torchvision.datasets.FashionMNIST(root=args.data_folder, train=False,
                                                    download=True, transform=transform_fmnist)

    elif args.dataset == 'cifar10dvs':
        num_cls = 10
        img_size = 64

        split_by = 'number'
        normalization = None
        T = args.num_steps  # number of frames

        dataset_dir = os.path.join(args.data_folder, args.dataset)
        if os.path.isdir(dataset_dir) is not True:
            os.mkdir(dataset_dir)

        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        # from spikingjelly.datasets import split_to_train_test_set  # Original function

        # Redefining split function to make it faster
        def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int,
                                    random_split: bool = False):
            '''
            :param train_ratio: split the ratio of the origin dataset as the train set
            :type train_ratio: float
            :param origin_dataset: the origin dataset
            :type origin_dataset: torch.utils.data.Dataset
            :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
            :type num_classes: int
            :param random_split: If ``False``, the front ratio of samples in each classes will
                    be included in train set, while the reset will be included in test set.
                    If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
                    ``numpy.randon.seed``
            :type random_split: int
            :return: a tuple ``(train_set, test_set)``
            :rtype: tuple
            '''
            import math
            label_idx = []

            if len(origin_dataset.samples) != 10000:  # If number of samples has been modified store label one by one
                for i in range(num_classes):
                    label_idx.append([])
                for i, item in enumerate(origin_dataset):
                    y = item[1]
                    if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
                        y = y.item()
                    label_idx[y].append(i)
            else:
                for i in range(10):  # Else, 1000 images per class
                    label_idx.append(list(range(i * 1000, (i + 1) * 1000)))
            train_idx = []
            test_idx = []
            if random_split:
                for i in range(num_classes):
                    np.random.shuffle(label_idx[i])

            for i in range(num_classes):
                pos = math.ceil(label_idx[i].__len__() * train_ratio)
                train_idx.extend(label_idx[i][0: pos])
                test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

            return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)

        origin_set = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T,
                                split_by=split_by)
        train_set, test_set = split_to_train_test_set(0.9, origin_set, 10)

    else:
        print("Dataset name not found")
        exit()

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)


    subset_percent = 0.9  # This should be a float value like 0.1 for 10%

    # Randomly select a subset from the training set
    num_subset = int(len(train_set) * subset_percent)
    subset_indices = torch.randperm(len(train_set)).tolist()[:num_subset]  # Random indices
    shadow_train_set = torch.utils.data.Subset(train_set, subset_indices)

    # Create a data loader for the shadow model
    shadow_trainloader = torch.utils.data.DataLoader(shadow_train_set, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=args.num_workers,
                                                    pin_memory=True, drop_last=True)
    num_groups=32
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
    if args.arch == "sresnet":
        model = SResnet(n=args.n, nFilters=args.nFilters, num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_classes, boosting=args.boosting, poisson_gen=args.poisson_gen, num_channels=num_channels).to(device)
    elif args.arch == "gsresnet":
        model = gSResnet(n=args.n, nFilters=args.nFilters, num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_classes, boosting=args.boosting, poisson_gen=args.poisson_gen, num_channels=num_channels).to(device)
    elif args.arch == "spikingvgg16":
         model = SpikingVGG16(num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_classes, boosting=args.boosting, poisson_gen=args.poisson_gen, num_channels=num_channels).to(device)
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")


    print(f"Dataset: {args.dataset}, Num Channels: {num_channels}, Num Classes: {num_classes}")
    best_acc = 0

    reload_epoch = None
    if args.reload is not None:

        model_dict = torch.load(args.reload, map_location='cpu')
        state_dict = model_dict['state_dict']
        reload_epoch = model_dict['global_step']
        best_acc = model_dict['accuracy']
        print('Reloading from epoch: ' + str(reload_epoch) + ' with Accuracy: ' + str(best_acc))

        if args.fine_tune:

            best_acc = 0
            own_state = model.state_dict()

            for name, param in state_dict.items():

                if 'conv_list.0' in name or 'conv1' in name or 'fc' in name:
                    continue

                own_state[name].copy_(param)
        else:
            model.load_state_dict(state_dict)

 
    target_model = model.to(device)
    # Configure the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    target_optimizer = optim.SGD(target_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    print('********** SNN simulation parameters **********')
    print('Simulation # time-step : {}'.format(num_steps))
    print('Membrane decay rate : {0:.2f}\n'.format(leak_mem))
    print('Poisson generation     : {}'.format(args.poisson_gen))

    print('********** Learning parameters **********')
    print('Backprop optimizer     : SGD')
    print('Batch size (training)  : {}'.format(batch_size))
    print('Batch size (testing)   : {}'.format(batch_size_test))
    print('Number of epochs       : {}'.format(num_epochs))
    print('Learning rate          : {}'.format(args.lr))

    print('********** Network architecture **********')
    print('Architecture           : SpikingVGG16')
    print('Group normalization    : {}'.format(num_groups))

    print('********** SNN training and evaluation **********')
    train_loss_list = []
    test_acc_list = []

    epoch = 0
    if reload_epoch is not None:
        epoch = reload_epoch
        reload_epoch = None
        if args.fine_tune:
            epoch = 0
        recalculate_learning_rate(target_optimizer, epoch, num_epochs, d1=0.7, d2=0.8, d3=0.9)

    while epoch <= num_epochs - 1:

        train_loss = AverageMeter()
        target_model.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)

            target_optimizer.zero_grad()
            output = target_model(inputs)

            loss = criterion(output, labels)

            train_loss.update(loss.item(), labels.size(0))

            loss.backward()
            target_optimizer.step()

            # Clear membrane potentials after each batch
            target_model.reset_membrane_potentials(inputs.size(0))

        if (epoch + 1) % args.train_display_freq == 0:
            print("Epoch: {}/{};".format(epoch + 1, num_epochs), "########## Training loss: {}".format(train_loss.avg))

        adjust_learning_rate(target_optimizer, epoch, num_epochs, d1=0.7, d2=0.8, d3=0.9)

        if (epoch + 1) % args.test_display_freq == 0:
            acc_top1, acc_top5 = [], []
            target_model.eval()
            with torch.no_grad():
                for j, data in enumerate(testloader, 0):
                    images, labels = data
                    images = images.to(device, dtype=torch.float)
                    labels = labels.to(device)
                    out = target_model(images)
                    prec1, prec5 = accuracy(out, labels, topk=(1, 5))
                    acc_top1.append(float(prec1))

                    # Clear membrane potentials after each batch
                    target_model.reset_membrane_potentials(images.size(0))

            test_accuracy = np.mean(acc_top1)
            print("test_accuracy : {}".format(test_accuracy))

            # Model save
            if best_acc < test_accuracy:
                best_acc = test_accuracy

                model_dict = {
                    'global_step': epoch + 1,
                    'state_dict': target_model.state_dict(),
                    'accuracy': test_accuracy
                }

                torch.save(model_dict, args.weight_folder + '/' + experiment_name + '_best.pth.tar')

        epoch += 1

    print('Completed training')
    print('Test accuracy: ' + str(test_accuracy))
    print('Best accuracy: ' + str(best_acc))

    # Train shadow model

    shadow_model = model.to(device)
    # Configure the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    shadow_optimizer = optim.SGD(shadow_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    print('********** SNN simulation parameters **********')
    print('Simulation # time-step : {}'.format(num_steps))
    print('Membrane decay rate : {0:.2f}\n'.format(leak_mem))
    print('Poisson generation     : {}'.format(args.poisson_gen))

    print('********** Learning parameters **********')
    print('Backprop optimizer     : SGD')
    print('Batch size (training)  : {}'.format(batch_size))
    print('Batch size (testing)   : {}'.format(batch_size_test))
    print('Number of epochs       : {}'.format(num_epochs))
    print('Learning rate          : {}'.format(args.lr))

    print('********** Network architecture **********')
    print('Architecture           : SpikingVGG16')
    print('Group normalization    : {}'.format(num_groups))

    print('********** SNN training and evaluation **********')
    train_loss_list = []
    test_acc_list = []

    epoch = 0
    if reload_epoch is not None:
        epoch = reload_epoch
        reload_epoch = None
        if args.fine_tune:
            epoch = 0
        recalculate_learning_rate(shadow_optimizer, epoch, num_epochs, d1=0.7, d2=0.8, d3=0.9)

    while epoch <= num_epochs - 1:

        train_loss = AverageMeter()
        shadow_model.train()
        for i, data in enumerate(shadow_trainloader):
            inputs, labels = data
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)

            shadow_optimizer.zero_grad()
            output = shadow_model(inputs)

            loss = criterion(output, labels)

            train_loss.update(loss.item(), labels.size(0))

            loss.backward()
            shadow_optimizer.step()

            # Clear membrane potentials after each batch
            shadow_model.reset_membrane_potentials(inputs.size(0))

        if (epoch + 1) % args.train_display_freq == 0:
            print("Epoch: {}/{};".format(epoch + 1, num_epochs), "########## Training loss: {}".format(train_loss.avg))

        adjust_learning_rate(shadow_optimizer, epoch, num_epochs, d1=0.7, d2=0.8, d3=0.9)

        if (epoch + 1) % args.test_display_freq == 0:
            acc_top1, acc_top5 = [], []
            shadow_model.eval()
            with torch.no_grad():
                for j, data in enumerate(testloader, 0):
                    images, labels = data
                    images = images.to(device, dtype=torch.float)
                    labels = labels.to(device)
                    out = shadow_model(images)
                    prec1, prec5 = accuracy(out, labels, topk=(1, 5))
                    acc_top1.append(float(prec1))

                    # Clear membrane potentials after each batch
                    shadow_model.reset_membrane_potentials(images.size(0))

            test_accuracy = np.mean(acc_top1)
            print("shadow test_accuracy : {}".format(test_accuracy))

            # Model save
            if best_acc < test_accuracy:
                best_acc = test_accuracy

                model_dict = {
                    'global_step': epoch + 1,
                    'state_dict': shadow_model.state_dict(),
                    'accuracy': test_accuracy
                }

                torch.save(model_dict, args.weight_folder + '/' + experiment_name + '_best.pth.tar')

        epoch += 1

    print('Completed training')
    print('shadow Test accuracy: ' + str(test_accuracy))
    print('shadow Best accuracy: ' + str(best_acc))


    from sklearn.svm import SVC

    # Assuming the shadow_model is already trained and available
    # Assuming shadow_train_loader and shadow_test_loader are defined and provide data in the CIFAR-10 format


    import numpy as np
    from sklearn.model_selection import train_test_split

    # Prepare the attack dataset
    num_classes = num_cls
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


    num_classes = num_cls
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
    roc_data_base_dir = "roc_data_v"

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
        plt.title(f'ROC Curve for Attack Model')
        plt.legend(loc="lower right")
        
        # Save the plot as an image file
        plot_file = os.path.join(dataset_dir, f'class_{i}_roc_curve.png')
        plt.savefig(plot_file)
        plt.close()

        print(f'Class {i} AUC: {roc_auc:.4f}')