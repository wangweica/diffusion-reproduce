import numpy as np
import torch.nn as nn
from myUtil import *
import cv2
import torch.nn.functional as F
from einops import rearrange
from inspect import isfunction
import math

class MyActivation(nn.Module):
    # 使用更适合INR的激活函数
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.sin(x)

class INRModel_v2(nn.Module):
    def __init__(self, channels=[], skip_to=[]):
        super(INRModel_v2, self).__init__()
        self.layers = nn.ModuleList()
        self.skip_to = skip_to

        for i in range(len(channels) - 1):
            if i in skip_to:
                self.layers.append(nn.Linear(channels[i]*2, channels[i + 1]))
            else:
                self.layers.append(nn.Linear(channels[i], channels[i + 1]))

            if i < len(channels) - 2:
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Sigmoid())
        # print(self.layers)

    def forward(self, x):
        layer_id = 0
        input_feature = x
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                layer_id += 1

                if layer_id in self.skip_to:
                    x = torch.cat((x, input_feature), dim=-1)

        return x

class GeneratedMLP(nn.Module):
    def __init__(self):
        super(GeneratedMLP, self).__init__()
        self.relu = nn.ReLU()
        self.params = []

    def set_parameters(self, params):
        self.params = params

    def forward(self, x):
        for layer in self.params:
            x = self.relu(torch.matmul(x, layer['weight']) + layer['bias'])
        return x
