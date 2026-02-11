from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.models.backbones.resnet.resnet_backbone import ResNetBackbone
from lib.models.backbones.hrnet.hrnet_backbone import HRNetBackbone
from lib.models.backbones.segformer.segformer_backbone import SegformerBackbone
from lib.models.backbones.mobilenet.mobilenet_v1 import MobileNetV1Backbone
from lib.models.backbones.mobilenet.mobilenet_v2 import MobileNetV2Backbone
from lib.models.backbones.mobilenet.mobilenet_v3 import MobileNetV3Backbone

from lib.utils.tools.logger import Logger as Log


class BackboneSelector(object):

    def __init__(self, configer):
        self.configer = configer

    def get_backbone(self, **params):
        backbone = self.configer.get('network', 'backbone') 

        model = None
        if ('resnet' in backbone or 'resnext' in backbone or 'resnest' in backbone) and 'senet' not in backbone:
            model = ResNetBackbone(self.configer)(**params)

        elif 'hrne' in backbone: 
            model = HRNetBackbone(self.configer)(**params)  

        elif 'segformer' in backbone:
            model = SegformerBackbone(self.configer)(**params)

        elif 'mobilenet_v1' in backbone:
            model = MobileNetV1Backbone(self.configer)(**params)
        elif 'mobilenet_v2' in backbone:
            model = MobileNetV2Backbone(self.configer)(**params)
        elif 'mobilenet_v3' in backbone:
            model = MobileNetV3Backbone(self.configer)(**params)

        else:
            Log.error('Backbone {} is invalid.'.format(backbone))
            exit(1)

        return model
