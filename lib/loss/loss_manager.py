from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.loss.cbl_loss import CBLLoss
from lib.loss.con_loss import CONLoss
from lib.loss.loss_helper import DiceLoss, BCELoss, GSCNNLoss, CELoss
from lib.utils.tools.logger import Logger as Log



SEG_LOSS_DICT = {
    'dice_loss': DiceLoss,
    'cbl_loss': CBLLoss,
    'bce_loss': BCELoss,
    'con_loss': CONLoss,
    'gscnn_loss':GSCNNLoss,
    'ce_loss': CELoss
}


class LossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def get_seg_loss(self, loss_type=None):
        key = loss_type
        if key not in SEG_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)
        loss = SEG_LOSS_DICT[key](self.configer) 

        return loss
