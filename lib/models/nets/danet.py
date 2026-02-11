import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper
from lib.models.modules.da_block import PAM_Module, CAM_Module



class DANet(nn.Module):
    def __init__(self, configer):
        super(DANet, self).__init__()

        self.configer = configer 
        self.num_classes = self.configer.get('data', 'num_classes')

        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 2048
        self.dahead = DANetHead(in_channels, self.num_classes, bn_type=self.configer.get('network', 'bn_type'))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        x = self.backbone(x)  
        x = self.dahead(x[-1])
        x = F.interpolate(x[0], size=(H, W), mode='bilinear', align_corners=True)
        out = self.sigmoid(x)
        return out
    


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, bn_type=None):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   ModuleHelper.BNReLU(inter_channels, bn_type=bn_type))
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   ModuleHelper.BNReLU(inter_channels, bn_type=bn_type))

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)

        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   ModuleHelper.BNReLU(inter_channels, bn_type=bn_type))
        
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   ModuleHelper.BNReLU(inter_channels, bn_type=bn_type))

        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)          # spatial attention
        sa_conv = self.conv51(sa_feat)    # feature
        # sa_output = self.conv6(sa_conv)   # aux output

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)          # channel attention
        sc_conv = self.conv52(sc_feat)    # feature
        # sc_output = self.conv7(sc_conv)   # aux output

        feat_sum = sa_conv+sc_conv             # feature add
        sasc_output = self.conv8(feat_sum)     # feature add output

        output = [sasc_output]
        # output.append(sa_output)
        # output.append(sc_output)
        return tuple(output)
