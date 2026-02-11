import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class PSPModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, pool_sizes=[1, 2, 3, 6], bn_type=None):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, bn_type) for size in pool_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(pool_sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size, bn_type):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = ModuleHelper.BNReLU(out_features, bn_type=bn_type)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle
