import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.tools.module_helper import ModuleHelper



class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
                )
            )     
            
    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = F.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            out_puts.append(ppm_out)
        return out_puts
 
    
class PPMHead(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes = [1, 2, 3, 6]):
        super(PPMHead, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes)*self.out_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out
 


 
class FPNHead(nn.Module):
    def __init__(self, channels=2048, out_channels=256, bn_type=None):
        super(FPNHead, self).__init__()
        self.PPMHead = PPMHead(in_channels=channels, out_channels=out_channels)
        
        self.Conv_fuse1 = nn.Sequential(
                nn.Conv2d(channels//2, out_channels, 1),
                ModuleHelper.BNReLU(out_channels, bn_type=bn_type)
            )
        self.Conv_fuse1_ = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1),
                ModuleHelper.BNReLU(out_channels, bn_type=bn_type)
            )
        self.Conv_fuse2 = nn.Sequential(
                nn.Conv2d(channels//4, out_channels, 1),
                ModuleHelper.BNReLU(out_channels, bn_type=bn_type)
            )    
        self.Conv_fuse2_ = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1),
                ModuleHelper.BNReLU(out_channels, bn_type=bn_type)
            )
        self.Conv_fuse3 = nn.Sequential(
                nn.Conv2d(channels//8, out_channels, 1),
                ModuleHelper.BNReLU(out_channels, bn_type=bn_type)
            ) 
        self.Conv_fuse3_ = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1),
                ModuleHelper.BNReLU(out_channels, bn_type=bn_type)
            )
        self.fuse_all = nn.Sequential(
                nn.Conv2d(out_channels*4, out_channels, 1),
                ModuleHelper.BNReLU(out_channels, bn_type=bn_type)
            )
 
    def forward(self, feature):
        x1 = self.PPMHead(feature[-1])
        f = x1

        x = self.Conv_fuse1(feature[-2])
        f = F.interpolate(f, size=x.shape[2:], mode='bilinear', align_corners=True)
        f = f + x
        x2 = self.Conv_fuse1_(f)

        x = self.Conv_fuse2(feature[-3])
        f = F.interpolate(f, size=x.shape[2:], mode='bilinear', align_corners=True)
        f = f + x
        x3 = self.Conv_fuse2_(f)  
 
        x = self.Conv_fuse3(feature[-4])
        f = F.interpolate(f, size=x.shape[2:], mode='bilinear', align_corners=True)
        f = f + x
        x4 = self.Conv_fuse3_(f)
 
        x1 = F.interpolate(x1, x4.shape[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, x4.shape[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, x4.shape[2:], mode='bilinear', align_corners=True)
 
        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))
        
        return x