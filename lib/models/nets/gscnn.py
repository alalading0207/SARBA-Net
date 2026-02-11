import torch
import torch.nn.functional as F
from torch import nn
from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper
from lib.models.modules.gscnn_block import BasicBlock, GatedSpatialConv2d, AtrousSpatialPyramidPoolingModule

import cv2
import numpy as np




class GSCNN(nn.Module):
    '''
    Wide_resnet version of DeepLabV3
    mod1
    pool2
    mod2 str2
    pool3
    mod3-7
      structure: [3, 3, 6, 3, 1, 1]
      channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                  (1024, 2048, 4096)]
    '''

    def __init__(self, configer):
        
        super(GSCNN, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.bn_type = self.configer.get('network', 'bn_type')
        wide_resnet = BackboneSelector(configer).get_backbone()
        
        # wide_resnet = wide_resnet.module
        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        self.interpolate = F.interpolate
        del wide_resnet

        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn7 = nn.Conv2d(4096, 1, 1)

        self.res1 = BasicBlock(64, 64, stride=1, downsample=None, bn_type=self.bn_type)
        self.d1 = nn.Conv2d(64, 32, 1)
        self.res2 = BasicBlock(32, 32, stride=1, downsample=None, bn_type=self.bn_type)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.res3 = BasicBlock(16, 16, stride=1, downsample=None, bn_type=self.bn_type)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = GatedSpatialConv2d(32, 32, bn_type=self.bn_type)
        self.gate2 = GatedSpatialConv2d(16, 16, bn_type=self.bn_type)
        self.gate3 = GatedSpatialConv2d(8, 8, bn_type=self.bn_type)
         
        self.aspp = AtrousSpatialPyramidPoolingModule(4096, 256, output_stride=8, bn_type=self.bn_type)

        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280 + 256, 256, kernel_size=1, bias=False)  # main branch aspp + boundary branch

        self.final_seg = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=self.bn_type),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=self.bn_type),
            nn.Conv2d(256, self.num_classes, kernel_size=1, bias=False))

        self.sigmoid = nn.Sigmoid()
        self.initialize_weights(self.final_seg)


    def forward(self, inp, canny):

        x_size = inp.size()    

        # res 1
        m1 = self.mod1(inp)    

        # res 2
        m2 = self.mod2(self.pool2(m1)) 

        # res 3
        m3 = self.mod3(self.pool3(m2)) 

        # res 4-7
        m4 = self.mod4(m3)  
        m5 = self.mod5(m4)  
        m6 = self.mod6(m5) 
        m7 = self.mod7(m6) 

        s3 = F.interpolate(self.dsn3(m3), x_size[2:], mode='bilinear', align_corners=True)    
        s4 = F.interpolate(self.dsn4(m4), x_size[2:], mode='bilinear', align_corners=True)   
        s7 = F.interpolate(self.dsn7(m7), x_size[2:], mode='bilinear', align_corners=True)    
        
        m1f = F.interpolate(m1, x_size[2:], mode='bilinear', align_corners=True)  

        # # canny
        # im_arr = inp.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8)  
        # canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))            
        # for i in range(x_size[0]):
        #     canny[i] = cv2.Canny(im_arr[i],10,100)
        # canny = torch.from_numpy(canny).cuda().float()   

        cs = self.res1(m1f)     
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)

        cs = self.d1(cs)        
        cs = self.gate1(cs, s3)  
        cs = self.res2(cs)      
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)

        cs = self.d2(cs)        
        cs = self.gate2(cs, s4)  
        cs = self.res3(cs)       
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)

        cs = self.d3(cs)         
        cs = self.gate3(cs, s7)  
        cs = self.fuse(cs)       
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)

        edge_out = self.sigmoid(cs)                 
        cat = torch.cat((edge_out, canny), dim=1)   
        acts = self.cw(cat)                        
        acts = self.sigmoid(acts)

        # aspp
        # x = self.aspp(m7, edge_out)   
        x = self.aspp(m7, acts)        
        dec0_up = self.bot_aspp(x)     
        dec0_up = self.interpolate(dec0_up, m2.size()[2:], mode='bilinear',align_corners=True)  

        dec0_fine = self.bot_fine(m2)  
        dec0 = [dec0_fine, dec0_up]  
        dec0 = torch.cat(dec0, 1)   

        dec1 = self.final_seg(dec0)  
        seg_out_con = self.interpolate(dec1, x_size[2:], mode='bilinear')            
        seg_out = self.sigmoid(seg_out_con)

        return seg_out_con, seg_out_con, edge_out
    


    def initialize_weights(self, *models):
        for model in models:
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()
