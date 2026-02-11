import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.modules.be_bc_block import BELModule, BEBModule, BCModule

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper
from lib.models.modules.projection import ProjectionHead
from lib.utils.tools.logger import Logger as Log


class HRNet(nn.Module):   # W48
    def __init__(self, configer):
        super(HRNet, self).__init__()

        self.configer = configer 
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_):
        H, W = x_.size(2), x_.size(3)

        x = self.backbone(x_)   
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)   
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)   
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)   


        feats = torch.cat([feat1, feat2, feat3, feat4], 1)    
        out = self.cls_head(feats)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True)
        out = self.sigmoid(out)
        return out
    

class HRNet_BE(nn.Module):
    def __init__(self, configer):
        super(HRNet_BE, self).__init__()
        
        self.configer = configer 
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.cbl_1_4 = BELModule(48, 48*5//2) 
        self.bce_1_4 = BEBModule(48*5//2, 1)   

        self.cbl_1_8 = BELModule(96, 96*5//2)
        self.bce_1_8 = BEBModule(96*5//2, 1)
        

    def forward(self, x_):
        H, W = x_.size(2), x_.size(3)
        
        x = self.backbone(x_)    
        _, _, h, w = x[0].size()

        # boundary enhancement module
        cbl_1_4 = self.cbl_1_4(x[0])
        bce_1_4 = self.bce_1_4(cbl_1_4)
        x[0] = bce_1_4*(x[0]+1)

        cbl_1_8 = self.cbl_1_8(x[1])
        bce_1_8 = self.bce_1_8(cbl_1_8)
        x[1] = bce_1_8*(x[1]+1)

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True) 


        feats = torch.cat([feat1, feat2, feat3, feat4], 1)    
        out = self.cls_head(feats)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True)
        # out = self.sigmoid(out)

        cbl = {'cbl_1_4': cbl_1_4, 'cbl_1_8': cbl_1_8}
        bce = {'bce_1_4': bce_1_4, 'bce_1_8': bce_1_8}

        return out, cbl, bce



class HRNet_BE_BC(nn.Module):
    def __init__(self, configer):
        super(HRNet_BE_BC, self).__init__()
        self.configer = configer 
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        self.att_1_4 = BCModule(48)
        self.cbl_1_4 = BELModule(48, 48*5//2) 
        self.bce_1_4 = BEBModule(48*5//2, 1)   

        self.att_1_8 = BCModule(96)
        self.cbl_1_8 = BELModule(96, 96*5//2)
        self.bce_1_8 = BEBModule(96*5//2, 1)
        

    def forward(self, x_):
        x = self.backbone(x_)    
        _, _, h, w = x[0].size()


        x0 = self.att_1_4(x[0])
        cbl_1_4 = self.cbl_1_4(x0)
        bce_1_4 = self.bce_1_4(cbl_1_4)
        x[0] = bce_1_4*(x[0]+1)

        x1 = self.att_1_8(x[1])
        cbl_1_8 = self.cbl_1_8(x1)
        bce_1_8 = self.bce_1_8(cbl_1_8)
        x[1] = bce_1_8*(x[1]+1)

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)     
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)     
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)     


        feats = torch.cat([feat1, feat2, feat3, feat4], 1)  
        out = self.cls_head(feats)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        con = out
        out = self.sigmoid(out)

        cbl = {'cbl_1_4': cbl_1_4, 'cbl_1_8': cbl_1_8}
        bce = {'bce_1_4': bce_1_4, 'bce_1_8': bce_1_8}

        return out, cbl, bce, con
