'''
Reproduction of the 1989 LeNet-5 model in PyTorch, adjusted for the modern MNIST dataset.
'''

import torch
import torch.nn as nn


class TorchLeNet89(nn.module):

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
		return x # TODO

	def param_count(self):
		print("Parameters: ", sum(p.numel() for p in self.parameters()))