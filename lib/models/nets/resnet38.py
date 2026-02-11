import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.modules.resnet38_block import ASPP, FCNHead
from lib.models.modules.be_bc_block import BELModule, BEBModule, BCModule
from lib.models.tools.module_helper import ModuleHelper



class ResNet38(nn.Module):    # wr38
    def __init__(self, configer):
        super(ResNet38, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        wide_resnet = BackboneSelector(configer).get_backbone()
        self.bn_type = self.configer.get('network', 'bn_type')

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

        self.aspp = ASPP(4096, 256, output_stride=8, bn_type=self.bn_type)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)
        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)

        self.final_seg = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=self.bn_type),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=self.bn_type),
            nn.Conv2d(256, self.num_classes, kernel_size=1, bias=False))

        self.sigmoid = nn.Sigmoid()
        self.initialize_weights(self.final_seg)


    def forward(self, inp):

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

        x = self.aspp(m7) 

        x = self.bot_aspp(x) 
        x =F.interpolate(x, size=(128,128), mode="bilinear", align_corners=True)

        dec0_fine = self.bot_fine(m2) 
        dec0_fine =F.interpolate(dec0_fine, size=(128,128), mode="bilinear", align_corners=True)
        dec0 = [dec0_fine, x]
        x = torch.cat(dec0, 1)

        x = self.final_seg(x)
        x =F.interpolate(x, size=x_size[2:], mode="bilinear", align_corners=True)
        # out = x
        out = self.sigmoid(x)

        return out
    

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
    



class ResNet38_BE(nn.Module):    # wr38
    def __init__(self, configer):
        super(ResNet38_BE, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        wide_resnet = BackboneSelector(configer).get_backbone()
        self.bn_type = self.configer.get('network', 'bn_type')

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

        self.aspp = ASPP(4096, 256, output_stride=8, bn_type=self.bn_type)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)
        self.bot_fine = nn.Conv2d(256, 48, kernel_size=1, bias=False)

        self.final_seg = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=self.bn_type),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=self.bn_type),
            nn.Conv2d(256, self.num_classes, kernel_size=1, bias=False))

        self.sigmoid = nn.Sigmoid()
        self.initialize_weights(self.final_seg)

        self.cbl_1_8 = BELModule(256, 256*5//4) 
        self.bce_1_8 = BEBModule(256*5//4, 1)  
 
        self.cbl_1_4 = BELModule(304, 304*5//4) 
        self.bce_1_4 = BEBModule(304*5//4, 1) 

    def forward(self, inp):

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

        x = self.aspp(m7)  
        x = self.bot_aspp(x) 
        x =F.interpolate(x, size=(32,32), mode="bilinear", align_corners=True)

        cbl_1_8 = self.cbl_1_8(x)
        bce_1_8 = self.bce_1_8(cbl_1_8)
        x = bce_1_8*(x+1)
        x =F.interpolate(x, size=(64,64), mode="bilinear", align_corners=True)

        dec0_fine = self.bot_fine(m3) 
        dec0_fine =F.interpolate(dec0_fine, size=(64,64), mode="bilinear", align_corners=True)
        dec0 = [dec0_fine, x]
        x = torch.cat(dec0, 1)

        cbl_1_4 = self.cbl_1_4(x)
        bce_1_4 = self.bce_1_4(cbl_1_4)
        x = bce_1_4*(x+1)


        x = self.final_seg(x)
        x =F.interpolate(x, size=x_size[2:], mode="bilinear", align_corners=True)
        # out = x
        out = self.sigmoid(x)

        cbl = {'cbl_1_4': cbl_1_4, 'cbl_1_8': cbl_1_8}
        bce = {'bce_1_4': bce_1_4, 'bce_1_8': bce_1_8}

        return out, cbl, bce
    

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




class ResNet38_BE_BC(nn.Module):    # wr38
    def __init__(self, configer):
        super(ResNet38_BE_BC, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        wide_resnet = BackboneSelector(configer).get_backbone()
        self.bn_type = self.configer.get('network', 'bn_type')

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

        self.aspp = ASPP(4096, 256, output_stride=8, bn_type=self.bn_type)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)
        self.bot_fine = nn.Conv2d(256, 48, kernel_size=1, bias=False)

        self.final_seg = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=self.bn_type),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=self.bn_type),
            nn.Conv2d(256, self.num_classes, kernel_size=1, bias=False))

        self.sigmoid = nn.Sigmoid()
        self.initialize_weights(self.final_seg)

        self.att_1_8 = BCModule(256)
        self.cbl_1_8 = BELModule(256, 256*5//4) 
        self.bce_1_8 = BEBModule(256*5//4, 1) 
 
        self.att_1_4 = BCModule(304)
        self.cbl_1_4 = BELModule(304, 304*5//4) 
        self.bce_1_4 = BEBModule(304*5//4, 1) 

    def forward(self, inp):

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

        x = self.aspp(m7)  
        x = self.bot_aspp(x) 
        x =F.interpolate(x, size=(32,32), mode="bilinear", align_corners=True)

        x = self.att_1_8(x)
        cbl_1_8 = self.cbl_1_8(x)
        bce_1_8 = self.bce_1_8(cbl_1_8)
        x = bce_1_8*(x+1)
        x =F.interpolate(x, size=(64,64), mode="bilinear", align_corners=True)

        dec0_fine = self.bot_fine(m3) 
        dec0_fine =F.interpolate(dec0_fine, size=(64,64), mode="bilinear", align_corners=True)
        dec0 = [dec0_fine, x]
        x = torch.cat(dec0, 1)

        x = self.att_1_4(x)
        cbl_1_4 = self.cbl_1_4(x)
        bce_1_4 = self.bce_1_4(cbl_1_4)
        x = bce_1_4*(x+1)


        x = self.final_seg(x)
        out =F.interpolate(x, size=x_size[2:], mode="bilinear", align_corners=True)
        con = out
        out = self.sigmoid(out)

        cbl = {'cbl_1_4': cbl_1_4, 'cbl_1_8': cbl_1_8}
        bce = {'bce_1_4': bce_1_4, 'bce_1_8': bce_1_8}

        return out, cbl, bce, con
    

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