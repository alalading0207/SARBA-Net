import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from lib.utils.tools.logger import Logger as Log




class DataHelper:

    def __init__(self, configer, trainer):
        self.configer = configer
        self.trainer = trainer


    def _prepare_sequence(self, seq, force_list=False):
        return self.trainer.module_runner.to_device_data(seq) 


    def prepare_data(self, data_dict):
        sequences = []
        for i in range(len(data_dict)):
            sequences.append(self._prepare_sequence(data_dict[i]))

        batch_size = len(data_dict[0])

        return sequences, batch_size 
    

    # large
    def prepare_data_larege(self, data_dict):
        sequences = []
        for i in range(len(data_dict)):
            sequences.append(self._prepare_sequence(data_dict[i]))

        batch_size = len(data_dict[0])

        return sequences, batch_size 
