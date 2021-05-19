import os
import torch.nn as nn
import torch
from torchvision import models

os.environ['TORCH_HOME'] = 'models'
# alexnet_model = models.alexnet(pretrained=True)
mobilenet_v3l_model = models.mobilenet_v3_large(pretrained=True)
resnet50_model = models.resnet50(pretrained=True)

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

class Resnet50PlusLatent(nn.Module):
    def __init__(self, n_classes, bits=128):
        super().__init__()
        self.resnet50 = resnet50_model
        self.bits = bits
        self.Linear1 = nn.Linear(2048, self.bits)
        self.softmax = nn.Softmax()
        self.Linear2 = nn.Linear(self.bits, n_classes)

    def forward(self, x, predict=False):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.resnet50.avgpool(x)
        x = x.view(x.size(0), -1)
        features = self.Linear1(x)
        if predict:
            return features
        features = self.softmax(features)
        result = self.Linear2(features)
        return features, result


class MobileNetV3LargePlusLatent(nn.Module):
    def __init__(self, n_classes, bits=128):
        super().__init__()
        self.bits = bits
        self.mobilenet = mobilenet_v3l_model
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.remain = nn.Sequential(*list(mobilenet_v3l_model.classifier.children())[:-1])
        self.Linear1 = nn.Linear(1280, self.bits)
        self.softmax = nn.Softmax()
        self.Linear2 = nn.Linear(self.bits, n_classes)

    def forward(self, x, predict=False):
        x = self.mobilenet.features(x)
        # x = x.view(x.size(0), 983040)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.remain(x)
        x = x.view(x.size(0), -1)
        features = self.Linear1(x)
        if predict:
            return features
        features = self.softmax(features)
        result = self.Linear2(features)
        return features, result


if __name__ == '__main__':
    Resnet50PlusLatent(13)