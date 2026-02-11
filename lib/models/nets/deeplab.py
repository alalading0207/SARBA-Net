import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.modules.deeplab_block import DeepLabHead, DeepLabHead_BE, ASPPModule, ASPPModule_BE
from lib.models.modules.be_bc_block import BELModule, BEBModule, BCModule
from lib.models.tools.module_helper import ModuleHelper



class DeepLabV3(nn.Module): 
    def __init__(self, configer):
        super(DeepLabV3, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        self.decoder = DeepLabHead(bn_type=self.configer.get('network', 'bn_type'))
        self.cls_head = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_):
        H, W = x_.size(2), x_.size(3)
        x = self.backbone(x_) 

        low_level = x[2]
        out = x[-1]
        x = self.decoder(low_level, out)

        x = self.cls_head(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        # out = x
        out = self.sigmoid(x)

        return out




class DeepLabV3_BE(nn.Module):    
    def __init__(self, configer):
        super(DeepLabV3_BE, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        self.bn_type = self.configer.get('network', 'bn_type')

        self.aspp = ASPPModule(2048, 256, bn_type=self.bn_type)

        self.bot_fine = nn.Sequential(nn.Conv2d(256, 48, kernel_size=3, stride=1, padding=1),
                                       ModuleHelper.BNReLU(48, bn_type=self.bn_type))

        self.cat_conv = nn.Sequential(
                                    nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
                                    ModuleHelper.BNReLU(256, bn_type=self.bn_type),
                                    nn.Dropout(0.1),
                                    nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                    ModuleHelper.BNReLU(256, bn_type=self.bn_type),
                                    nn.Dropout(0.1)
                                    )
        self.cls_head = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.initialize_weights(self.cat_conv)

        self.cbl_1_8 = BELModule(256, 256*5//4)  
        self.bce_1_8 = BEBModule(256*5//4, 1)   

        self.cbl_1_4 = BELModule(256, 256*5//4) 
        self.bce_1_4 = BEBModule(256*5//4, 1)       


    def forward(self, x_):
        H, W = x_.size(2), x_.size(3)

        x_ = self.backbone(x_)  
        x = self.aspp(x_[-1])  
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=True) 

        cbl_1_8 = self.cbl_1_8(x)
        bce_1_8 = self.bce_1_8(cbl_1_8)
        x = bce_1_8*(x+1)

        dec0_fine = x_[2]
        dec0_fine = self.bot_fine(dec0_fine) 
        dec0_fine = F.interpolate(dec0_fine, size=(64, 64), mode='bilinear', align_corners=True) 

        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True) 
        dec0 = [dec0_fine, x]
        x = torch.cat(dec0, 1)
        x = self.cat_conv(x)

        
        cbl_1_4 = self.cbl_1_4(x)
        bce_1_4 = self.bce_1_4(cbl_1_4)
        x = bce_1_4*(x+1)

        x = self.cls_head(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
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



class DeepLabV3_BE_BC(nn.Module):    
    def __init__(self, configer):
        super(DeepLabV3_BE_BC, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        self.bn_type = self.configer.get('network', 'bn_type')

        self.aspp = ASPPModule(2048, 256, bn_type=self.bn_type)

        self.bot_fine = nn.Sequential(nn.Conv2d(256, 48, kernel_size=3, stride=1, padding=1),
                                       ModuleHelper.BNReLU(48, bn_type=self.bn_type))

        self.cat_conv = nn.Sequential(
                                    nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
                                    ModuleHelper.BNReLU(256, bn_type=self.bn_type),
                                    nn.Dropout(0.1),
                                    nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                    ModuleHelper.BNReLU(256, bn_type=self.bn_type),
                                    nn.Dropout(0.1)
                                    )
        self.cls_head = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.initialize_weights(self.cat_conv)

        self.att_1_8 = BCModule(256)
        self.cbl_1_8 = BELModule(256, 256*5//4)  
        self.bce_1_8 = BEBModule(256*5//4, 1)   

        self.att_1_4 = BCModule(256)
        self.cbl_1_4 = BELModule(256, 256*5//4) 
        self.bce_1_4 = BEBModule(256*5//4, 1)           


    def forward(self, x_):
        H, W = x_.size(2), x_.size(3)

        x_ = self.backbone(x_) 
        x = self.aspp(x_[-1]) 
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=True) 

        x = self.att_1_8(x)
        cbl_1_8 = self.cbl_1_8(x)
        bce_1_8 = self.bce_1_8(cbl_1_8)
        x = bce_1_8*(x+1)

        dec0_fine = x_[2]
        dec0_fine = self.bot_fine(dec0_fine)  
        dec0_fine = F.interpolate(dec0_fine, size=(64, 64), mode='bilinear', align_corners=True) 

        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True) 
        dec0 = [dec0_fine, x]
        x = torch.cat(dec0, 1)
        x = self.cat_conv(x)

        x = self.att_1_4(x)
        cbl_1_4 = self.cbl_1_4(x)
        bce_1_4 = self.bce_1_4(cbl_1_4)
        x = bce_1_4*(x+1)

        x = self.cls_head(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        con = x
        # out = x
        out = self.sigmoid(x)

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

