import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper
from lib.models.modules.uper_block import FPNHead



class UPerNet(nn.Module):
    def __init__(self, configer):
        super(UPerNet, self).__init__()

        self.configer = configer 
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()


        in_channels = 2048
        channels = 256

        self.decoder = FPNHead(in_channels, channels, bn_type=self.configer.get('network', 'bn_type'))

        self.cls_head = nn.Conv2d(channels, self.num_classes, kernel_size=1)
        # self.cls_head = nn.Sequential(
        #     nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        #     ModuleHelper.BNReLU(channels, bn_type=self.configer.get('network', 'bn_type')),
        #     nn.Dropout2d(0.10),
        #     nn.Conv2d(channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        # )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        H, W = x.size(2), x.size(3)

        x = self.backbone(x)   
        x = self.decoder(x)

        x = self.cls_head(x)
        x = F.interpolate(x, size=(H, W),mode='bilinear', align_corners=True)
        out = self.sigmoid(x)
        return out