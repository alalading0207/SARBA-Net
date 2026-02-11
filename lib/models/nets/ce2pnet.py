import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper
from lib.models.modules.edge_block import Edge_Module
from lib.models.modules.ce2p_block import CE2P_Decoder_Module
from lib.models.modules.psp_block import PSPModule




class CE2PNet(nn.Module):
    def __init__(self, configer):
        super(CE2PNet, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        
        in_channels = [1024, 2048]
        self.edge_layer = Edge_Module(256, self.num_classes, bn_type=self.configer.get('network', 'bn_type'), factor=1) 

        self.psp_layer = PSPModule(features=2048, 
                        out_features=512, 
                        bn_type=self.configer.get('network', 'bn_type')
                        )
        self.decoder = CE2P_Decoder_Module(self.num_classes, 
                        dropout=0.1, 
                        bn_type=self.configer.get('network', 'bn_type'),
                        inplane1=512,
                        inplane2=256
                        )

        self.cls = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                ModuleHelper.BNReLU(256, bn_type=self.configer.get('network', 'bn_type')),
                nn.Dropout2d(0.10),
                nn.Conv2d(256, self.num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
            )
        self.sigmoid = nn.Sigmoid()
       

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x = self.backbone(x)   # x: list output from conv2_x, conv3_x, conv4_x, conv5_x

        # edge branch
        edge_out, edge_fea = self.edge_layer(x[-4], x[-3], x[-2]) 

        # main branch
        x_hr = self.psp_layer(x[-1])
        seg_out1, x_hr = self.decoder(x_hr, x[-4])  
        
        # fuse branch
        x_hr = torch.cat([x_hr, edge_fea], dim=1)
        seg_out2 = self.cls(x_hr)

    
        seg_out1 = F.interpolate(seg_out1, size=(H, W), mode="bilinear", align_corners=True)
        seg_out2 = F.interpolate(seg_out2, size=(H, W), mode="bilinear", align_corners=True)
        edge_out = F.interpolate(edge_out, size=(H, W), mode="bilinear", align_corners=True)

        seg_out1 = self.sigmoid(seg_out1)
        seg_out2 = self.sigmoid(seg_out2)
        edge_out = self.sigmoid(edge_out)

        return seg_out1, seg_out2, edge_out

