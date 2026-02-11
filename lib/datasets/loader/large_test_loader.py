from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import pdb

import numpy as np
from torch.utils import data

import torch
import torch.nn.functional as F

from lib.utils.helpers.image_helper import ImageHelper
from lib.utils.tools.logger import Logger as Log



# loader = klass(root_dir, **kwargs)
class TestLoader_large(data.Dataset):
    def __init__(self, root_dir,  mode=None, img_transform=None, label_transform=None, edge_transform=None, configer=None):   # dataset='train'
        self.configer = configer
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.edge_transform = edge_transform
        self.img_list, self.name_list = self.__list_dirs(root_dir, mode)
        Log.info('{} {} \n'.format(mode, len(self.img_list))) 


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, index):

        name = self.name_list[index]

        # read image (4096, 4096, 1)
        img = ImageHelper.read_image(self.img_list[index],   
                                     tool=self.configer.get('data', 'image_tool'),   # cv2
                                     mode=self.configer.get('data', 'input_mode'))   # BGR


        large_input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # (4096, 4096, 1) -> torch.Size([1, 1, 4096, 4096])
        patches = F.unfold(large_input_tensor, kernel_size=(256, 256), stride=(128, 128)) # torch.Size([1, 65536, 961])
        num_patches = patches.shape[-1]     # 961

        patches = patches.permute(0, 2, 1)  # torch.Size([1, 961, 65536])
        patches = patches.view(-1, img.shape[2], 256, 256)  # torch.Size([*, 1, 256, 256])

        patches = patches.permute(0, 2, 3, 1)
        patches = patches.numpy()       # (961, 256, 256, 1)


        if self.img_transform:
            patches = [self.img_transform(patch) for patch in patches]
            patches = torch.stack(patches)  # torch.Size([961, 1, 256, 256])

        return patches, num_patches, name
    
    

    def __list_dirs(self, root_dir, mode):
        img_list = list()
        name_list = list()
        image_dir = root_dir


        dataset = self.configer.get('dataset')      # Wuhan_4096
        reader=open(f'/home/dyl/boundary/boundary_c/lib/datasets/name_list/{dataset}/{mode}.txt','r')
        files_name=reader.read().splitlines()  

        for file_name in files_name:

            img_path = os.path.join(image_dir, '{}'.format(file_name))  


            if not os.path.exists(img_path):
                Log.error('IMAGE Path: {} not exists.'.format(img_path))
                continue

            img_list.append(img_path)
            name_list.append(file_name)

        return img_list, name_list
    


if __name__ == "__main__":
    pass
