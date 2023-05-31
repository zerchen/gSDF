#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File :conv_head.py
#@Date :2022/09/27 21:16:21
#@Author :zerui chen
#@Contact :zerui.chen@inria.fr


import torch
import torch.nn as nn

class ConvHead(nn.Module):
    def __init__(self, feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
        super().__init__()
        layers = []
        for i in range(len(feat_dims)-1):
            layers.append(nn.Conv2d(in_channels=feat_dims[i], out_channels=feat_dims[i + 1], kernel_size=kernel, stride=stride, padding=padding))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers) 

    def forward(self, inp):
        return self.layers(inp)
