import logging
import cv2
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from lib.models.modules.decouple_block import SqueezeBodyEdge
from lib.models.tools.module_helper import ModuleHelper
from lib.models.backbones.backbone_selector import BackboneSelector



class Decouple(nn.Module):
    """
    Implement DeepFCN model
    Note that our FCN model is a strong baseline that
    A: stride8
    B: stride16
    with skip connections
    """
    def __init__(self, configer, skip='m1', skip_num=48):
        super(Decouple, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        self.bn_type = self.configer.get('network', 'bn_type')


        self.fcn_head = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=self.bn_type)
        )

        self.bot_fine = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        

        # body edge generation module
        self.squeeze_body_edge = SqueezeBodyEdge(256, bn_type=self.bn_type)

        # fusion different edge part
        self.edge_fusion = nn.Conv2d(256 + 48, 256, 1, bias=False)
        self.edge_out = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(48, bn_type=self.bn_type),
            nn.Conv2d(48, 1, kernel_size=1, bias=False)
        )

        # DSN for seg body part
        self.dsn_seg_body = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=self.bn_type),
            nn.Conv2d(256, self.num_classes , kernel_size=1, bias=False)
        )

        # Final segmentation part
        self.final_seg = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=self.bn_type),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=self.bn_type),
            nn.Conv2d(256, self.num_classes , kernel_size=1, bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        out_size = inp.size()[2:]  # 256,256
        
        x = self.backbone(inp)

        # use default low-level feature
        dec0_fine = self.bot_fine(x[2])   # 16,48,65,65
        fine_size = dec0_fine.size()[2:]  # 65,65

        # main branch
        aspp = self.fcn_head(x[-1])  
        
        # decouple
        seg_body, seg_edge = self.squeeze_body_edge(aspp)

 
        # fusion fine to edge == seg edge out
        seg_edge = F.interpolate(seg_edge, size=fine_size, mode='bilinear', align_corners=True)  
        seg_edge = self.edge_fusion(torch.cat([seg_edge, dec0_fine], dim=1))  
        seg_edge_out = self.edge_out(seg_edge)

        # seg body out
        seg_body = F.interpolate(seg_body, size=fine_size, mode='bilinear', align_corners=True)  
        seg_body_out = self.dsn_seg_body(seg_body)


        # fusion edge to body == segout
        seg_out = seg_edge + seg_body

        # fusion main_branch to segout
        aspp = F.interpolate(aspp, size=fine_size, mode='bilinear', align_corners=True)   
        seg_out = torch.cat([aspp, seg_out], dim=1)
        seg_final = self.final_seg(seg_out)

        # output
        seg_edge_out = F.interpolate(seg_edge_out, size=out_size, mode='bilinear', align_corners=True)   
        seg_edge_out = self.sigmoid(seg_edge_out)

        seg_final_out = F.interpolate(seg_final, size=out_size, mode='bilinear', align_corners=True)  
        seg_final_out = self.sigmoid(seg_final_out)

        seg_body_out = F.interpolate(seg_body_out, size=out_size, mode='bilinear', align_corners=True)   
        seg_body_out = self.sigmoid(seg_body_out)


        return seg_final_out, seg_body_out, seg_edge_out