import torch
from torch import nn
from torch.nn import functional as F

from lib.models.tools.module_helper import ModuleHelper


class SEModule(nn.Module):
    """Squeeze and Extraction module"""

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module based on DeepLab v3 settings"""

    def __init__(self, in_dim, out_dim, d_rate=[12, 24, 36], bn_type=None):
        super(ASPPModule, self).__init__()
        self.b0 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1,
                                          bias=False),
                                ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
                                )
        self.b1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3,
                                          padding=d_rate[0],
                                          dilation=d_rate[0], bias=False),
                                ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
                                )
        self.b2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3,
                                          padding=d_rate[1],
                                          dilation=d_rate[1], bias=False),
                                ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
                                )
        self.b3 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3,
                                          padding=d_rate[2],
                                          dilation=d_rate[2], bias=False),
                                ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
                                )
        self.b4 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(in_dim, out_dim, kernel_size=1,
                                          padding=0, bias=False),
                                ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
                                )

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
        )

    def forward(self, x):
        h, w = x.size()[2:]
        feat0 = self.b0(x)   
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = F.interpolate(self.b4(x), size=(h, w), mode='bilinear',
                              align_corners=True)

        out = torch.cat((feat0, feat1, feat2, feat3, feat4), dim=1) 
        return self.project(out)



class ASPPModule_BE(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''
    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18], bn_type=None):
        super(ASPPModule_BE, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          ModuleHelper.BNReLU(reduction_dim, bn_type=bn_type)
                          ))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3, dilation=r, padding=r, bias=False),
                ModuleHelper.BNReLU(reduction_dim, bn_type=bn_type)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            ModuleHelper.BNReLU(reduction_dim, bn_type=bn_type))
       
    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)           # 16,4096,1,1
        img_features = self.img_conv(img_features)   # 16,256,1,1
        img_features = F.interpolate(img_features, x_size[2:], mode='bilinear', align_corners=True) 
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out 



class DeepLabHead(nn.Module):
    """Segmentation head based on DeepLab v3"""

    def __init__(self, bn_type=None):
        super(DeepLabHead, self).__init__()
        # short
        self.layer_short = nn.Sequential(nn.Conv2d(256, 48, kernel_size=3,
                                                 stride=1, padding=1),
                                       ModuleHelper.BNReLU(48, bn_type=bn_type)
                                       )
        # main pipeline
        self.layer_aspp = ASPPModule(2048, 256, bn_type=bn_type)

        self.cat_conv = nn.Sequential(
                                    nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
                                    ModuleHelper.BNReLU(256, bn_type=bn_type),
                                    nn.Dropout(0.5),

                                    nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                    ModuleHelper.BNReLU(256, bn_type=bn_type),
                                    nn.Dropout(0.1),
                                    )
        

    def forward(self, low, high):
        b,c,h,w = low.shape

        low = self.layer_short(low)         # 16,256,65,65  -->   16,48,65,65
        low = F.interpolate(low, size=(h, w), mode='bilinear', align_corners=True)

        high_aspp = self.layer_aspp(high)  
        high_aspp = F.interpolate(high_aspp, size=(h, w), mode='bilinear', align_corners=True)  
        
        x = self.cat_conv(torch.cat((high_aspp, low), dim=1))

        return x
    

class DeepLabHead_BE(nn.Module):
    def __init__(self, bn_type=None):
        super(DeepLabHead_BE, self).__init__()
        # short
        self.layer_short = nn.Sequential(nn.Conv2d(256, 48, kernel_size=3,
                                                 stride=1, padding=1),
                                       ModuleHelper.BNReLU(48, bn_type=bn_type),
                                       )
        # main pipeline
        self.layer_aspp = ASPPModule(2048, 256, bn_type=bn_type)

        self.cat_conv = nn.Sequential(
                                    nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
                                    ModuleHelper.BNReLU(256, bn_type=bn_type),
                                    nn.Dropout(0.5),

                                    nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                    ModuleHelper.BNReLU(256, bn_type=bn_type),
                                    nn.Dropout(0.1),
                                    )
        

    def forward(self, low, high):
        b,c,h,w = low.shape

        low = self.layer_short(low)        
        low = F.interpolate(low, size=(h-1, w-1), mode='bilinear', align_corners=True) 

        high_aspp = self.layer_aspp(high)  
        high_aspp = F.interpolate(high_aspp, size=(h-1, w-1), mode='bilinear', align_corners=True) 
        
        x = self.cat_conv(torch.cat((high_aspp, low), dim=1))

        return x






class DeepLabHead_MobileNet_V1(nn.Module):
    """Segmentation head based on DeepLab v3"""

    def __init__(self, num_classes, bn_type=None):
        super(DeepLabHead_MobileNet_V1, self).__init__()
        # main pipeline
        self.layer_aspp = ASPPModule(1024, 512, bn_type=bn_type)
        self.refine = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,
                                              padding=1, stride=1, bias=False),
                                    ModuleHelper.BatchNorm2d(bn_type=bn_type)(512),
                                    nn.Conv2d(512, num_classes, kernel_size=1,
                                              stride=1, bias=True))

    def forward(self, x):
        # aspp module
        x_aspp = self.layer_aspp(x)
        # refine module
        x_seg = self.refine(x_aspp)

        return x_seg

class DeepLabHead_MobileNet_V3(nn.Module):
    """Segmentation head based on DeepLab v3"""

    def __init__(self, num_classes, bn_type=None):
        super(DeepLabHead_MobileNet_V3, self).__init__()
        # main pipeline
        self.layer_aspp = ASPPModule(960, 512, bn_type=bn_type)
        self.refine = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,
                                              padding=1, stride=1, bias=False),
                                    ModuleHelper.BatchNorm2d(bn_type=bn_type)(512),
                                    nn.Conv2d(512, num_classes, kernel_size=1,
                                              stride=1, bias=True))

    def forward(self, x):
        # aspp module
        x_aspp = self.layer_aspp(x)
        # refine module
        x_seg = self.refine(x_aspp)

        return x_seg

class DeepLabHead_MobileNet(nn.Module):
    """Segmentation head based on DeepLab v3"""

    def __init__(self, num_classes, bn_type=None):
        super(DeepLabHead_MobileNet, self).__init__()
        # main pipeline
        self.layer_aspp = ASPPModule(1280, 512, bn_type=bn_type)
        self.refine = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,
                                              padding=1, stride=1, bias=False),
                                    ModuleHelper.BatchNorm2d(bn_type=bn_type)(512),
                                    nn.Conv2d(512, num_classes, kernel_size=1,
                                              stride=1, bias=True))

    def forward(self, x):
        # aspp module
        x_aspp = self.layer_aspp(x)
        # refine module
        x_seg = self.refine(x_aspp)

        return x_seg

