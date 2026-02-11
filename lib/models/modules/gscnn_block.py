import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import numpy as np
import math
from lib.models.tools.module_helper import ModuleHelper



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_type=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 =  ModuleHelper.BN(planes, bn_type=bn_type)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 =ModuleHelper.BN(planes, bn_type=bn_type)

        self.downsample = downsample
        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class AtrousSpatialPyramidPoolingModule(nn.Module):
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
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

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
            ModuleHelper.BNReLU(reduction_dim, bn_type=bn_type)
            )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            ModuleHelper.BNReLU(reduction_dim, bn_type=bn_type)
            )
         

    def forward(self, x, edge): 
        x_size = x.size()

        img_features = self.img_pooling(x)            # 16,4096,1,1
        img_features = self.img_conv(img_features)    # 16,256,1,1
        img_features = F.interpolate(img_features, x_size[2:], mode='bilinear',align_corners=True)
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:], mode='bilinear',align_corners=True)
        edge_features = self.edge_conv(edge_features)   
        out = torch.cat((out, edge_features), 1)         

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)    
        return out                         




class GatedSpatialConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, bn_type=None):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        """

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialConv2d, self).__init__( 
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self._gate_conv = nn.Sequential(
            ModuleHelper.BN(in_channels+1, bn_type=bn_type),
            nn.Conv2d(in_channels+1, in_channels+1, 1),
            nn.ReLU(), 
            nn.Conv2d(in_channels+1, 1, 1),
            ModuleHelper.BN(1, bn_type=bn_type),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        """

        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))  

        input_features = (input_features * (alphas + 1)) 
        return F.conv2d(input_features, self.weight, self.bias, self.stride, 
                        self.padding, self.dilation, self.groups)  
  


