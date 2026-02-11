from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.modules.bfp_block import UAG_RNN
from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper




class BFP(nn.Module):
    def __init__(self, configer):
        super(BFP, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')

        self.backbone = BackboneSelector(configer).get_backbone()
        self.head = BFPHead(2048, self.num_classes, bn_type=self.configer.get('network', 'bn_type'))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x =  self.backbone(x)

        x = self.head(x[-1]) 
        x = list(x)

        x[0] = F.interpolate(x[0], size=(H, W), mode="bilinear", align_corners=True)
        x[1] = F.interpolate(x[1], size=(H, W), mode="bilinear", align_corners=True)

        return self.sigmoid(x[0]), x[1] 
        # return x[0], x[1] 
        


class BFPHead(nn.Module):
    def __init__(self, in_channels, out_channels, bn_type=None):
        super(BFPHead, self).__init__()
        inter_channels = in_channels // 4
        self.no_class=out_channels+1
        self.adapt1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   ModuleHelper.BNReLU(inter_channels, bn_type=bn_type))
        
        self.adapt2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, dilation=12 , padding=12, bias=False),
                                   ModuleHelper.BNReLU(inter_channels, bn_type=bn_type))
        
        self.adapt3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, dilation=12 , padding=12, bias=False),
                                   ModuleHelper.BNReLU(inter_channels, bn_type=bn_type))
        
        self.uag_rnn = UAG_RNN(inter_channels)

        self.seg1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   ModuleHelper.BNReLU(inter_channels, bn_type=bn_type),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(512, out_channels+2, 1))
        
        self.seg2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   ModuleHelper.BNReLU(inter_channels, bn_type=bn_type),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(512, out_channels, 1))                                 
        
        self.softmax = nn.Softmax(dim=1)
        self.gamma = nn.Parameter(2*torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1)/out_channels)

    def forward(self, x):
        # adapt from CNN
        feat1 = self.adapt1(x)
        feat2 = self.adapt2(feat1)
        # Boundary
        s1_output = self.seg1(feat2) 
        s1_output_ = self.softmax(s1_output)
        score_ = torch.narrow(s1_output, 1, 0, self.no_class)  
        boundary_ = torch.narrow(s1_output_, 1, self.no_class, 1) 

        ## boundary confidence to propagation confidence, method 2
        boundary = torch.mean(torch.mean(boundary_, 2, True), 3, True)-boundary_+self.bias 
        boundary = (boundary - torch.min(torch.min(boundary, 3, True)[0], 2, True)[0])*self.gamma

        boundary = torch.clamp(boundary, max=1)
        boundary = torch.clamp(boundary, min=0)
        ## UAG-RNN
        feat3 = self.adapt3(feat1)
        uag_feat = self.uag_rnn(feat3, boundary)
        feat_sum = uag_feat + feat3 #residual
        s2_output = self.seg2(feat_sum)
        # sd_output = self.conv7(sd_conv)

        # output1 = s2_output + score_   
        output1 = s2_output + torch.narrow(score_, 1, 1, 1) 


        output = [output1]
        output.append(s1_output)

        return tuple(output)

