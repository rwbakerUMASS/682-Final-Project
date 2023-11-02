import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,pred,true):
        return torch.mean(torch.log(torch.cosh(pred - true + 1e-12)))