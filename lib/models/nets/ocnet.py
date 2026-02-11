import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class BaseOCNet(nn.Module):
    """
    OCNet: Object Context Network for Scene Parsing
    """
    def __init__(self, configer):
        self.inplanes = 128
        super(BaseOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        in_channels = [1024, 2048]
        self.oc_module_pre = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.base_oc_block import BaseOC_Module
        self.oc_module = BaseOC_Module(in_channels=512, 
                                       out_channels=512,
                                       key_channels=256, 
                                       value_channels=256,
                                       dropout=0.05, 
                                       sizes=([1]),
                                       bn_type=self.configer.get('network', 'bn_type'))
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_):
        H, W = x_.size(2), x_.size(3)

        x = self.backbone(x_)
        x = self.oc_module_pre(x[-1])
        x = self.oc_module(x)
        x = self.cls(x)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
        out = self.sigmoid(x)
        return out


class AspOCNet(nn.Module):
    """
    OCNet: Object Context Network for Scene Parsing
    """
    def __init__(self, configer):
        self.inplanes = 128
        super(AspOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        in_channels = [1024, 2048]
        from lib.models.modules.asp_oc_block import ASP_OC_Module
        self.context = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            ASP_OC_Module(512, 256, bn_type=self.configer.get('network', 'bn_type')),
        )
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_):
        H, W = x_.size(2), x_.size(3)

        x = self.backbone(x_)
        x = self.context(x[-1])
        x = self.cls(x)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
        out = self.sigmoid(x)
        return out
