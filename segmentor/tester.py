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



class Tester(object):
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
        self.test_loader = self.seg_data_loader.get_testloader()
        self.test_size = len(self.test_loader) * self.configer.get('test', 'batch_size')



    def test(self, data_loader=None):
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

            if self.configer.exists('other_boundary') and self.configer.get('other_boundary', 'name')=="gscnn":
                (img, canny, labelmap, _), batch_size= self.data_helper.prepare_data(data_dict)
                with torch.no_grad():
                    output = self.gscnn_test(img, canny, labelmap) 

                    g_hat = self.con_out(output)
                    g_hat_ = g_hat.clone().permute(0, 2, 3, 1).cpu().detach().numpy()
                    for k in range(batch_size):
                        image_id += 1                    
                        label_img = np.squeeze((g_hat_[k]*255).astype(np.uint8)) 
                        label_img_ = Image.fromarray(label_img)
                        name = (name_list[k]).split(".")[0] 
                        label_path = os.path.join(self.save_dir, '{}.jpg'.format(name))
                        ImageHelper.save(label_img_, label_path) 

            else:
                (img, labelmap), batch_size= self.data_helper.prepare_data(data_dict)
                with torch.no_grad():
                    output = self.ss_test(img, labelmap) 


                    # output pred
                    output_ = torch.round(output.clone()).long()
                    output_ = output_.permute(0, 2, 3, 1).cpu().detach().numpy()
                    for k in range(batch_size):
                        image_id += 1                    
                        label_img = np.squeeze((output_[k]*255).astype(np.uint8)) 
                                        # .astype(np.unit8)
                        label_img_ = Image.fromarray(label_img)
                        name = (name_list[k]).split(".")[0]   #
                        label_path = os.path.join(self.save_dir, '{}.jpg'.format(name))
                        ImageHelper.save(label_img_, label_path)  

                    # # output boundary
                    # g_hat = self.con_out(output)
                    # g_hat_ = g_hat.clone().permute(0, 2, 3, 1).cpu().detach().numpy()
                    # for k in range(batch_size):
                    #     image_id += 1                    
                    #     label_img = np.squeeze((g_hat_[k]*255).astype(np.uint8)) 
                    #     label_img_ = Image.fromarray(label_img)
                    #     name = (name_list[k]).split(".")[0]   
                    #     label_path = os.path.join(self.save_dir, '{}.jpg'.format(name))
                    #     ImageHelper.save(label_img_, label_path) 
                    

            Log.info('iters_id:{}'.format(j)) 
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        # Print the log info & reset the states.
        Log.info('Test Time {batch_time.sum:.3f}s'.format(batch_time=self.batch_time))

        
    

    def ss_test(self, img, labelmap):

        start = timeit.default_timer()
        if not self.configer.exists("boundary") and not self.configer.exists("other_boundary"):
            output = self.seg_net(img)

        elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_be'):
            output, _, _ = self.seg_net(img)

        elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_bc'):
            output, _ = self.seg_net(img)

        elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_be_bc'):
            output, _, _, _ = self.seg_net(img)

        elif self.configer.exists('other_boundary') and self.configer.get('other_boundary', 'name')=="ce2p":
            _, output, _ = self.seg_net(img)

        elif self.configer.exists('other_boundary') and self.configer.get('other_boundary', 'name')=="decouple":
            output, _, _ = self.seg_net(img)

        elif self.configer.exists('other_boundary') and self.configer.get('other_boundary', 'name')=="bfp":
            output, _ = self.seg_net(img)
            
        else:
            raise argparse.ArgumentTypeError('Unsupported test mode...')

        self.evaluator.update_score(output, labelmap)
        self.evaluator.update_test()
        end = timeit.default_timer()
        
        return output
    


    def gscnn_test(self, img, canny, labelmap):

        start = timeit.default_timer()
        output, _, _ = self.seg_net(img, canny)

        self.evaluator.update_score(output, labelmap)
        self.evaluator.update_test()
        end = timeit.default_timer()
        
        return output


if __name__ == "__main__":
    pass