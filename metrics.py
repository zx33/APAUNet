# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

import numpy as np
import torch
from medpy import metric
import torch.nn.functional as F


def load_dicefunc(output, target):
    smooth = 1e-4
    if torch.is_tensor(output):
        output = torch.nn.Softmax(dim=1)(output)
        output = torch.argmax(output, 1)
        output = output.data.cpu().numpy()

    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    dice = [[], [], []]
    for idx in range(output.shape[0]):
        for i in range(3):
            intersection = ((output[idx] == i) & (target[idx] == i)).sum()
            union = ((output[idx] == i) | (target[idx] == i)).sum()
            cof = (intersection + smooth) / (union + smooth)
            dice[i].append(2 * cof / (1 + cof))
    return np.asarray(dice)

def load_hdfunc(outputs, targets):
    output = 0 * outputs[:,0] + 1 * outputs[:,1] + 2 * outputs[:,2]
    output = output.unsqueeze(1)
    hd95 = metric.binary.hd95(output.detach().cpu().numpy(), targets.detach().cpu().numpy())
    return hd95
