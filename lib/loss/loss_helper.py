from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from lib.utils.tools.logger import Logger as Log



class DiceLoss(nn.Module):
    def __init__(self, configer):
        super(DiceLoss, self).__init__()
        self.smooth = 1
        self.epsilon = 1e-8

    def forward(self, pred, true):
        intersection = 2. * torch.sum(pred * true) + self.epsilon
        union = torch.sum(pred) + torch.sum(true) + self.epsilon
        score = intersection / union
        loss = 1-score
        return loss
    

class BCELoss(nn.Module):
    def __init__(self, configer):
        super(BCELoss, self).__init__()
        self.loss = nn.BCELoss()
        
    def forward(self, pred, true):
        loss = self.loss(pred, true)
        return loss
    

class GSCNNLoss(nn.Module):
    def __init__(self, configer):
        super(GSCNNLoss, self).__init__()
        self.loss = nn.BCELoss()
        
    def forward(self, pred, true, edge):
        filler = torch.ones_like(true)
        loss = self.loss(pred, torch.where(edge.max(1)[0].unsqueeze(1) > 0.8, true, filler))  
        return loss


class CELoss(nn.Module):
    def __init__(self, configer):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, pred, true):
        loss = self.loss(pred, true.long())  
        return loss