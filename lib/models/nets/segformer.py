import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.modules.be_bc_block import BELModule, BEBModule, BCModule
from lib.utils.tools.logger import Logger as Log


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs  # c1[16,64,64,64]   c2[16,128,32,32]   c3[16,320,16,16]   c4[16,512,8,8]

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])   
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])  
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

    

class Segformer(nn.Module):
    def __init__(self, configer):
        super(Segformer, self).__init__()
        self.configer = configer 
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        arch = self.configer.get('network', 'backbone')
        if 'b0' in arch:
            self.in_channels = [32, 64, 160, 256]
            self.embedding_dim = 256

        elif 'b1' in arch:
            self.in_channels = [64, 128, 320, 512]
            self.embedding_dim = 256

        elif 'b2' in arch or 'b3' in arch or 'b4' in arch or 'b5' in arch:
            self.in_channels = [64, 128, 320, 512]
            self.embedding_dim = 768
        else:
            Log.error('Backbone phase {} is invalid.'.format(arch))
            exit(1)

        self.decode_head = SegFormerHead(self.num_classes, self.in_channels, self.embedding_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        
        x = self.backbone.forward(inputs)  
        x = self.decode_head.forward(x)
        
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.sigmoid(x)
        return x
    









class SegFormerHead_BE(nn.Module):
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)

        self.cbl_1_4 = BELModule(64, 64*5//2) 
        self.bce_1_4 = BEBModule(64*5//2, 1)       
        self.cbl_1_8 = BELModule(128, 128*5//2)
        self.bce_1_8 = BEBModule(128*5//2, 1)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs 

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        cbl_1_4 = self.cbl_1_4(c1)
        bce_1_4 = self.bce_1_4(cbl_1_4)
        c1 = bce_1_4*(c1+1)

        cbl_1_8 = self.cbl_1_8(c2)
        bce_1_8 = self.bce_1_8(cbl_1_8)
        c2 = bce_1_8*(c2+1)
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) 
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3]) 
        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3]) 

        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

    

class Segformer_BE(nn.Module):
    def __init__(self, configer):
        super(Segformer, self).__init__()
        self.configer = configer 
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        arch = self.configer.get('network', 'backbone')
        if 'b0' in arch:
            self.in_channels = [32, 64, 160, 256]
            self.embedding_dim = 256

        elif 'b1' in arch:
            self.in_channels = [64, 128, 320, 512]
            self.embedding_dim = 256

        elif 'b2' in arch or 'b3' in arch or 'b4' in arch or 'b5' in arch:
            self.in_channels = [64, 128, 320, 512]
            self.embedding_dim = 768
        else:
            Log.error('Backbone phase {} is invalid.'.format(arch))
            exit(1)

        self.decode_head = SegFormerHead_BE(self.num_classes, self.in_channels, self.embedding_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        
        x = self.backbone.forward(inputs)  
        x = self.decode_head.forward(x)
        
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.sigmoid(x)
        return x