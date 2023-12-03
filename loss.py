import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import torchvision.models as models

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,pred,true):
        return torch.mean(torch.log(torch.cosh(pred - true + 1e-12)))
    
vgg_model = models.vgg16(pretrained=True).features

# Freeze the layers of the VGG16 model
for param in vgg_model.parameters():
    param.requires_grad = False

# Define perceptual loss as L2 norm between feature maps
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg_layers = vgg_model.to('cuda')[:23] # Use layers until the 4th max-pooling layer (before fully connected layers)
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        x_features = self.vgg_layers(x)
        y_features = self.vgg_layers(y)
        loss = self.criterion(x_features, y_features)
        return loss
