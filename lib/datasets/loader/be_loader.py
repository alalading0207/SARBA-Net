from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import pdb
import torch
import argparse

import numpy as np
from torch.utils import data

from lib.utils.helpers.image_helper import ImageHelper
from lib.utils.tools.logger import Logger as Log



# loader = klass(root_dir, **kwargs)
class BELoader(data.Dataset):
    def __init__(self, root_dir,  mode=None, img_transform=None, label_transform=None, edge_transform=None, configer=None):   # dataset='train'
        self.configer = configer
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.edge_transform = edge_transform
        self.img_list, self.label_list = self.__list_dirs(root_dir, mode) 
        Log.info('{} {} \n'.format(mode, len(self.img_list))) 


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, index):

        if self.configer.exists('boundary') and self.configer.get('boundary', 'use_be'):
            resolution_list = self.configer.get('boundary', 'resolution')

            # read image
            img = ImageHelper.read_image(self.img_list[index], 
                                        tool=self.configer.get('data', 'image_tool'),   # cv2
                                        mode=self.configer.get('data', 'input_mode'))   # BGR
            # read label
            labelmap = ImageHelper.read_image(self.label_list[index],
                                            tool=self.configer.get('data', 'image_tool'),
                                            mode=self.configer.get('data', 'input_mode')) 
            

            # To tensor / deal image
            img = self.img_transform(img)  

            if 1 in resolution_list:
                edge = self.edge_transform(labelmap)

            # To tensor / deal label
            labelmap = self.label_transform(labelmap)  



            sub_boundary = []
            for i, resolution in enumerate(resolution_list):  # "use_be": [2,4,8]   [1,2,4,8]   [1,4,8] ...
                if resolution==1 :
                    sub_boundary.append(edge)
                else:
                    kernel = stride = [resolution, resolution]
                    sub_boundary.append(self.get_downsample_boundary(labelmap, kernel, stride, 1/resolution))

        else:
            
            raise argparse.ArgumentTypeError('Wrong Boundary setting...')
        
        return img, labelmap, sub_boundary
                

    def get_downsample_boundary(self, input, kernel_size, stride, lower_bound, upper_bound=1):
        boundary = torch.nn.functional.avg_pool2d(input, kernel_size, stride)
        boundary = (boundary>lower_bound) & (boundary < upper_bound)
        return boundary.float()


    def __list_dirs(self, root_dir, mode):

        img_list = list()
        label_list = list()
        image_dir = os.path.join(root_dir, 'image')
        label_dir = os.path.join(root_dir, 'label')

        dataset = self.configer.get('dataset')
        reader=open(f'/home/dyl/boundary/boundary_c/lib/datasets/name_list/{dataset}/{mode}.txt','r')
        files_name=reader.read().splitlines()

        for file_name in files_name:

            img_path = os.path.join(image_dir, '{}'.format(file_name)) 
            label_path = os.path.join(label_dir, '{}'.format(file_name)) 

            if not os.path.exists(label_path) or not os.path.exists(img_path):
                Log.error('Label Path: {} {} not exists.'.format(label_path, img_path))
                continue

            img_list.append(img_path)
            label_list.append(label_path)

        return img_list, label_list
    
    

if __name__ == "__main__":
    pass
