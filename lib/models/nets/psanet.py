"""Point-wise Spatial Attention Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.modules.psa_block import PSAHead
from lib.models.modules.fcn_head import FCNHead



class PSANet(nn.Module):
    def __init__(self, configer):
        super(PSANet, self).__init__()

        self.configer = configer 
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        self.aux=False

        self.psahead = PSAHead(self.num_classes, bn_type=self.configer.get('network', 'bn_type'))
        self.sigmoid = nn.Sigmoid()

        if self.aux:
            self.auxlayer = FCNHead(1024, self.num_classes)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x = self.backbone(x) 
        x = self.psahead(x[-1])
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        out = self.sigmoid(x)
        return out






