from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.utils.tools.logger import Logger as Log



# Boumdary series Net
from lib.models.nets.ce2pnet import CE2PNet
from lib.models.nets.gscnn import GSCNN
from lib.models.nets.bfp import BFP
from lib.models.nets.decouple import Decouple

# Seg series Net
from lib.models.nets.upernet import UPerNet
from lib.models.nets.danet import DANet
from lib.models.nets.pspnet import PSPNet
from lib.models.nets.psanet import PSANet
from lib.models.nets.ocnet import BaseOCNet, AspOCNet
from lib.models.nets.resnet38 import ResNet38, ResNet38_BE, ResNet38_BE_BC

# UNet 
from lib.models.nets.unet import UNet, UNet_BE, UNet_BC, UNet_BE_BC, UNet_BE_1, UNet_BC_1, UNet_BE_BC_1

# HRNet
from lib.models.nets.hrnet import HRNet, HRNet_BE, HRNet_BE_BC

# Deeplabv3+
from lib.models.nets.deeplab import DeepLabV3, DeepLabV3_BE, DeepLabV3_BE_BC

# Segformer
from lib.models.nets.segformer import Segformer, Segformer_BE



SEG_MODEL_DICT = {

    # Boumdary series Net
    'ce2pnet': CE2PNet,
    'gscnn': GSCNN,
    'bfp': BFP,
    'decouple': Decouple,


    # Seg series Net
    'upernet': UPerNet,
    'danet': DANet,
    'psanet': PSANet,
    'pspnet': PSPNet,
    'ocnet': BaseOCNet,
    # 'ocnet': AspOCNet,


    # Resnet38 series
    'resnet38':ResNet38,
    'resnet38_be':ResNet38_BE,
    'resnet38_be_bc':ResNet38_BE_BC,

    # HRNet series
    'hrnet': HRNet,
    'hrnet_be': HRNet_BE,
    'hrnet_be_bc': HRNet_BE_BC,

    # Segformer series
    'segformer': Segformer,
    'segformer_be': Segformer_BE,

    # Deeplab series
    'deeplab': DeepLabV3,
    'deeplab_be': DeepLabV3_BE,
    'deeplab_be_bc': DeepLabV3_BE_BC,

    # UNet series
    'unet': UNet,
    'unet_be': UNet_BE,
    'unet_bc': UNet_BC,
    'unet_be_bc': UNet_BE_BC,

    'unet_be_1': UNet_BE_1,
    'unet_bc_1': UNet_BC_1,
    'unet_be_bc_1': UNet_BE_BC_1


}


class ModelManager(object):
    def __init__(self, configer):
        self.configer = configer

    def semantic_segmentor(self):
        model_name = self.configer.get('network', 'model_name')  # 获取model name

        if model_name not in SEG_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = SEG_MODEL_DICT[model_name](self.configer)

        return model
