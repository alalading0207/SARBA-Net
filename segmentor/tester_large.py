from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import timeit
import pdb
import cv2
import scipy
import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.helpers.file_helper import FileHelper
from lib.utils.helpers.image_helper import ImageHelper
from lib.utils.tools.average_meter import AverageMeter
from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.tools.logger import Logger as Log
from lib.vis.seg_visualizer import SegVisualizer
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.evaluator import get_evaluator
from segmentor.tools.data_helper import DataHelper
from PIL import Image
from lib.loss.con_loss import CON_out



class Tester_large(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """

    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.seg_visualizer = SegVisualizer(configer)
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.seg_data_loader = DataLoader(configer)
        self.data_helper = DataHelper(configer, self)
        self.evaluator = get_evaluator(configer, self)
        self.save_dir = self.configer.get('test', 'out_dir')
        self.con_out = CON_out()
        
        self.seg_net = None
        self.test_loader = None
        self.test_size = None
        self.name_list = None
        self.infer_time = 0
        self.infer_cnt = 0
        self._init_model()

    def _init_model(self):
        # net
        self.seg_net = self.model_manager.semantic_segmentor()
        self.seg_net = self.module_runner.load_net(self.seg_net)

        # dataset
        self.test_loader = self.seg_data_loader.get_testloader_large()
        self.test_size = len(self.test_loader) * self.configer.get('test', 'batch_size')



    def test_large(self, data_loader=None):
        """
          Validation function during the train phase.
        """
        self.seg_net.eval()
        start_time = time.time()
        image_id = 0

        Log.info('save dir {}'.format(self.save_dir))
        FileHelper.make_dirs(self.save_dir, is_file=False)

        data_loader = self.test_loader if data_loader is None else data_loader
        for j, data_dict in enumerate(data_loader):
            name_list = data_dict.pop()
            num_patches = data_dict.pop()

            # prepare data
            patches, batch_size= self.data_helper.prepare_data(data_dict) # patches: list: [1, 961, 1, 256, 256]
            patches = patches[0][0]     # torch.Size([961, 1, 256, 256])


            all_preds=[]
            with torch.no_grad():
                # pred
                for i in range(0, num_patches, 32):
                    batch_patches = patches[i:i+32]  
                    outputs = self.ss_test(batch_patches)  
                    all_preds.append(outputs)    # 31  torch.Size([32, 1, 256, 256])  


                # cat
                all_preds = torch.cat(all_preds, dim=0)  # (961, 1, 256, 256)
                all_preds = all_preds.permute(1, 2, 3, 0)             # (1, 256, 256, num_patches)
                all_preds = all_preds.contiguous().view(1, 1 * 256 * 256, -1)  # (bs, 1*256*256, num_patches)
                large = F.fold(all_preds, output_size=(4096,4096), kernel_size=(256, 256), stride=(128, 128))


                # output
                large_ = torch.round(large.clone()).long()
                large_ = large_.permute(0, 2, 3, 1).cpu().detach().numpy()  # (1, 4096, 4096, 1)
                for k in range(batch_size):
                    image_id += 1                    
                    label_img = np.squeeze((large_[k]*255).astype(np.uint8)) 
                    label_img_ = Image.fromarray(label_img)

                    name = (name_list[k]).split(".")[0] 
                    label_path = os.path.join(self.save_dir, '{}.tif'.format(name))
                    ImageHelper.save(label_img_, label_path) 


                print("Success:", j)


            Log.info('iters_id:{}'.format(j)) 
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        # Print the log info & reset the states.
        Log.info('Test Time {batch_time.sum:.3f}s'.format(batch_time=self.batch_time))


    

    def ss_test(self, img):

        start = timeit.default_timer()
        if not self.configer.exists("boundary") and not self.configer.exists("other_boundary"):
            output = self.seg_net(img)

        elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_be'):
            output, _, _ = self.seg_net(img)

        elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_bc'):
            output, _ = self.seg_net(img)

        elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_be_bc'):
            output, _, _, _ = self.seg_net(img)
            
        else:
            raise argparse.ArgumentTypeError('Unsupported test mode...')

        end = timeit.default_timer()
        
        return output
    




if __name__ == "__main__":
    pass