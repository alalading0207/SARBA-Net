import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.tools.module_helper import ModuleHelper



class ASPP(nn.Module):
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
        super(ASPP, self).__init__()

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




class FCNHead(nn.Module):
    def __init__(self, in_channels, channels, bn_type=None):   # in_channels=1024  channels=64
        super(FCNHead, self).__init__()

        self.up8 = nn.Sequential(
                     nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
                     ModuleHelper.BNReLU(256, bn_type=bn_type)
        )

        self.up4 = nn.Sequential(
                     nn.Conv2d(256, 128, 3, padding=1, bias=False),
                     ModuleHelper.BNReLU(128, bn_type=bn_type)
        )

        self.up2 = nn.Sequential(
                     nn.Conv2d(128, channels, 3, padding=1, bias=False),
                     ModuleHelper.BNReLU(channels, bn_type=bn_type)
        )

    def forward(self, x):
        x =F.interpolate(x, size=(64,64), mode="bilinear", align_corners=True)
        x = self.up8(x)
        x =F.interpolate(x, size=(128,128), mode="bilinear", align_corners=True)
        x = self.up4(x)
        x =F.interpolate(x, size=(256,256), mode="bilinear", align_corners=True)
        x = self.up2(x)

        return x