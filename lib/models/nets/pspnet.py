import torch
import torch.nn.functional as F
from torch import nn

from lib.models.modules.psp_block import PSPModule
from lib.models.backbones.backbone_selector import BackboneSelector


class PSPNet(nn.Module):
    def __init__(self, configer):
        super(PSPNet, self).__init__()

        self.configer = configer 
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 2048   # psp input channels

        self.psp_branch = nn.Sequential(
            PSPModule(in_channels, pool_sizes=[1, 2, 3, 6], bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(in_channels//4,  self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
            )
        self.sigmoid = nn.Sigmoid()
        self.initialize_weights(self.psp_branch)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x = self.backbone(x)
        output = self.psp_branch(x[-1])
        out = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)
        out = self.sigmoid(out)
        return out

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()