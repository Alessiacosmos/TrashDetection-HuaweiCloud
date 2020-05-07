# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2020/5/7 15:59
@Author  : Alessia K
@Email   : ------
"""

"""
build your model at this file

TIPS:
pytorch==1.0.0
torchvision is lower than 0.2.0 


(some module such as 'torchvision.ops' is not supported when update your model to evaluation in Huawei ModelArts, 
please change your code to fit the version)
"""

import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


class YourNetStruct(nn.Module):
    def __init__(self, num_classes, block, layers):
        super(YourNetStruct, self).__init__()
        pass

    def forward(self, x):
        return x



def yournetclass(numclass=80, pretrained=False):
    model = models.resnet18()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model