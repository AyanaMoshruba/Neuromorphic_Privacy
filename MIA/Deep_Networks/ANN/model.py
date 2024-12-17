import torch.nn as nn

def compute_num_groups(channels):
    # Example logic: set groups as square root of channels if possible, otherwise default to 1
    possible_groups = [8, 4, 2, 1]  # You can adjust this list as per your preferences
    for g in possible_groups:
        if channels % g == 0:
            return g
    return 1
# Define the BasicBlock for the ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        num_groups1 = compute_num_groups(out_channels)
        num_groups2 = compute_num_groups(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups1, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups2, out_channels)
        
        if stride != 1 or in_channels != out_channels:
            num_groups_shortcut = compute_num_groups(out_channels)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups_shortcut, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += residual
        out = self.relu(out)
        return out
    

# Define the ResNet-6 model
class ResNet6(nn.Module):
    def __init__(self, num_classes, num_channels=3):
        super(ResNet6, self).__init__()
        self.in_channels = 16  # Initial number of channels
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, 16)  # Assuming we use 4 groups for 16 channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(BasicBlock, 16, 2, stride=1)
        self.layer2 = self.make_layer(BasicBlock, 32, 2, stride=2)
        self.layer3 = self.make_layer(BasicBlock, 64, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

class ResNet18(nn.Module):
    def __init__(self, num_classes=100, num_channels=3):
        super(ResNet18, self).__init__()
        self.in_channels = 64  # Increased initial channels
        self.conv1 = nn.Conv2d(num_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, self.in_channels)  
        self.relu = nn.ReLU(inplace=True)
        
        # Increasing the number of blocks for added capacity
        self.layer1 = self.make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)  # Adjusted for 100 classes

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # New layer
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        
        # First 1x1 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_channels)
        
        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        # Third 1x1 convolution (this expands the channel dimension)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(8, out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()

        # Adjust the shortcut path to match the increase in channels and potential downsample
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels * self.expansion)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)
        
        out += residual
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes, num_channels=3):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(num_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Adjust block counts for ResNet50: [3, 4, 6, 3]
        self.layer1 = self.make_layer(Bottleneck, 256, 3, stride=1)  # out_channels = 256 as bottleneck's last conv expands the channels
        self.layer2 = self.make_layer(Bottleneck, 512, 4, stride=2)
        self.layer3 = self.make_layer(Bottleneck, 1024, 6, stride=2)
        self.layer4 = self.make_layer(Bottleneck, 2048, 3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8192, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion  # Make sure to account for the expansion factor in Bottleneck blocks
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        print(out.shape)
        out = self.layer2(out)
        print(out.shape)
        out = self.layer3(out)
        print(out.shape)
        out = self.layer4(out) 
        print(out.shape)# New layer
        out = self.avg_pool(out)
        print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes=1000, num_channels=3, num_groups=32):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 128),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 256),
            nn.ReLU(inplace=True))
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 256),
            nn.ReLU(inplace=True))
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 512),
            nn.ReLU(inplace=True))
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 512),
            nn.ReLU(inplace=True))
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 512),
            nn.ReLU(inplace=True))
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 512),
            nn.ReLU(inplace=True))
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1*1*512, 4096),  # Adjusted for 32x32 input
            nn.ReLU(inplace=True))
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True))
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
