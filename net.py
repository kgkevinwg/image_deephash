import os
import torch.nn as nn
import torch
from torchvision import models

os.environ['TORCH_HOME'] = 'models'
# alexnet_model = models.alexnet(pretrained=True)
mobilenet_v3l_model = models.mobilenet_v3_large(pretrained=True)


# class AlexNetPlusLatent(nn.Module):
#     def __init__(self, n_classes, bits=128):
#         super(AlexNetPlusLatent, self).__init__()
#         self.bits = bits
#         self.features = nn.Sequential(*list(alexnet_model.features.children()))
#         self.remain = nn.Sequential(*list(alexnet_model.classifier.children())[:-1])
#         self.Linear1 = nn.Linear(4096, self.bits)
#         self.sigmoid = nn.Sigmoid()
#         self.Linear2 = nn.Linear(self.bits, n_classes)
#
#     def forward(self, x, predict=False):
#         x = self.features(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         x = self.remain(x)
#         x = self.Linear1(x)
#         features = self.sigmoid(x)
#         if predict:
#             return features
#         result = self.Linear2(features)
#         return features, result


class MobileNetV3LargePlusLatent(nn.Module):
    def __init__(self, n_classes, bits=128):
        super().__init__()
        self.bits = bits
        self.features = nn.Sequential(*list(mobilenet_v3l_model.features.children()))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.remain = nn.Sequential(*list(mobilenet_v3l_model.classifier.children())[:-1])
        self.Linear1 = nn.Linear(1280, self.bits)
        self.sigmoid = nn.Sigmoid()
        self.Linear2 = nn.Linear(self.bits, n_classes)

    def forward(self, x, predict=False):
        x = self.features(x)
        # x = x.view(x.size(0), 983040)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.remain(x)
        x = x.view(x.size(0), -1)
        x = self.Linear1(x)
        features = self.sigmoid(x)
        if predict:
            return features
        result = self.Linear2(features)
        return features, result
