import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.tools.module_helper import ModuleHelper




class PSAHead(nn.Module):
    def __init__(self, nclass, bn_type=None):
        super(PSAHead, self).__init__()
        # psa_out_channels = crop_size // 8 ** 2
        self.psa = _PointwiseSpatialAttention(2048, 3600, bn_type)

        self.conv_post =nn.Sequential(
                nn.Conv2d(1024, 2048, 1),
                ModuleHelper.BNReLU(2048, bn_type=bn_type)
            )
        
        self.project = nn.Sequential(
                nn.Conv2d(4096, 512, 3, padding=1),
                ModuleHelper.BNReLU(512, bn_type=bn_type),
                nn.Dropout2d(0.1, False),
                nn.Conv2d(512, nclass, 1)
            )

    def forward(self, x):
        global_feature = self.psa(x)
        out = self.conv_post(global_feature)
        out = torch.cat([x, out], dim=1)
        out = self.project(out)

        return out


class _PointwiseSpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, bn_type=None):
        super(_PointwiseSpatialAttention, self).__init__()
        reduced_channels = 512
        self.collect_attention = _AttentionGeneration(in_channels, reduced_channels, out_channels, bn_type)
        self.distribute_attention = _AttentionGeneration(in_channels, reduced_channels, out_channels, bn_type)

    def forward(self, x):
        collect_fm = self.collect_attention(x)
        distribute_fm = self.distribute_attention(x)
        psa_fm = torch.cat([collect_fm, distribute_fm], dim=1)
        return psa_fm


class _AttentionGeneration(nn.Module):
    def __init__(self, in_channels, reduced_channels, out_channels, bn_type):
        super(_AttentionGeneration, self).__init__()
        self.conv_reduce = nn.Sequential(
                nn.Conv2d(in_channels, reduced_channels, 1),
                ModuleHelper.BNReLU(reduced_channels, bn_type=bn_type)
            )

        self.attention = nn.Sequential(
                nn.Conv2d(reduced_channels, reduced_channels, 1),
                ModuleHelper.BNReLU(reduced_channels, bn_type=bn_type),
                nn.Conv2d(reduced_channels, out_channels, 1, bias=False)
            )

        self.reduced_channels = reduced_channels

    def forward(self, x):
        reduce_x = self.conv_reduce(x) 
        attention = self.attention(reduce_x)    
        n, c, h, w = attention.size()   
        attention = attention.view(n, c, -1) 
        reduce_x = reduce_x.view(n, self.reduced_channels, -1) 
        fm = torch.bmm(reduce_x, torch.softmax(attention, dim=1))

        fm = fm.view(n, self.reduced_channels, h, w)   

        return fm