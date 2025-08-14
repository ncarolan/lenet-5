'''
Implementation of the LeNet-5 image classification model in PyTorch.
'''

import torch
import torch.nn as nn

class TorchLeNet(nn.Module):
    """ LeNet-5 in PyTorch adapted to 28x28 pixel inputs (MNIST).  """

    def __init__(self, act_fn: str, init: str):
        super().__init__()

        self.act_fn = act_fn
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        # TODO: Add dropout

        self._init_weights(init)
        self.act_fn = self._get_act_fn(act_fn)

    def forward(self, x):
        x = self.pool1(torch.sigmoid(self.conv1(x)))
        x = self.pool2(torch.sigmoid(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.fc3(x)
        return x

    def _init_weights(self, init: str) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if init == 'xavier':        nn.init.xavier_uniform_(m.weight)
                elif init == 'kaiming':     nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif init == 'orthogonal':  nn.init.orthogonal_(m.weight)
                else:                       nn.init.normal_(m.weight) # Default to normal distribution
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _get_act_fn(self, act_fn: str) -> nn.Module:
        if act_fn == "relu":
            return nn.ReLU()
        elif act_fn == "tanh":
            return nn.Tanh()
        elif act_fn == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")

    def param_count(self):
        print("Parameters: ", sum(p.numel() for p in self.parameters()))