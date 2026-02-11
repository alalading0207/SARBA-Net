from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import numpy as np
import torch


from lib.utils.tools.logger import Logger as Log
from lib.loss.con_loss import CON_metric


class RunningScore(object):

    def __init__(self, configer):
        self.configer = configer
        self.epsilon = 1e-8
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

        self.con_metric = CON_metric()
        self.count = 0
        self.mae = 0

    def update(self, pred, true):
        pred = torch.round(pred).long() 
        self.TP += torch.sum(pred * true)
        self.FP += torch.sum(pred * (1-true))
        self.TN += torch.sum((1-pred) * (1-true))
        self.FN += torch.sum((1-pred) * true)

        self.count += 1
        self.mae +=  self.con_metric(pred.float(), true)

    def Accuracy(self):
        accuracy = (self. TP + self.TN) / (self. TP + self.FP + self.TN + self.FN)
        return accuracy.detach().cpu().numpy()

    def Precision(self):
        precision = self. TP  / (self. TP + self.FP )
        return precision.detach().cpu().numpy()
    
    def Recall(self):
        recall = self. TP / (self. TP + self.FN)
        return recall.detach().cpu().numpy()

    def F1_score(self):
        precision = self.Precision()
        recall = self.Recall()
        return 2 / (1 / precision + 1 / recall + self.epsilon)

    def Iou(self):
        iou = self. TP  / (self. TP + self.FP  + self.FN)
        return iou.detach().cpu().numpy()

    def MIou(self):
        iou = self.Iou()
        iou_ = self. TN  / (self. TN + self.FP  + self.FN)
        miou = (iou + iou_.detach().cpu().numpy())/2
        return miou
    
    def Kappa(self):
        total = self. TP + self.FP + self.TN + self.FN
        accuracy =  (self. TP + self.TN) / total
        p1 = (self. TP + self.FN) * (self. TP + self.FP)
        p2 = (self. FP + self.TN) * (self. TN + self.FN)
        pe = (p1 + p2) / (total * total)
        kappa = (accuracy - pe) / (1 - pe)
        return kappa.detach().cpu().numpy()
    
    def Boundary(self):
        boundary_metric = self.mae / (self.count + self.epsilon)
        return boundary_metric.detach().cpu().numpy()
    
    def reset(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0