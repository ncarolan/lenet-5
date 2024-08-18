'''
Implementations of the LeNet-5 image classification model in PyTorch.
'''

import torch
import torch.nn as nn

class TorchLeNet(nn.Module):
    """ LeNet-5 in PyTorch adapted to 28x28 pixel inputs (modern MNIST).  """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool1(torch.sigmoid(self.conv1(x)))
        x = self.pool2(torch.sigmoid(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x) # Add softmax?
        return x

    def param_count(self):
        print("Parameters: ", sum(p.numel() for p in self.parameters()))


class TorchLeNet89(nn.Module):
    """ LeNet-5 in PyTorch with original 16x16 pixel inputs (classic MNIST).  """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def param_count(self):
        print("Parameters: ", sum(p.numel() for p in self.parameters()))