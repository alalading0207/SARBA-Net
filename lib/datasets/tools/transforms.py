from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch
from PIL import Image


class Normalize(object):

    def __init__(self, div_value, mean, std):
        self.div_value = div_value
        self.mean = mean
        self.std =std

    def __call__(self, inputs):
        inputs = inputs.div(self.div_value) 
        for t, m, s in zip(inputs, self.mean, self.std):
            t.sub_(m).div_(s)

        return inputs


class ToTensor(object):

    def __call__(self, inputs):
        inputs = torch.Tensor(inputs.transpose(2, 0, 1))
        return inputs.float()
    

class ToLabel(object):
    def __init__(self, div_value):
        self.div_value = div_value

    def __call__(self, inputs):
        inputs = inputs.div(self.div_value)
        return inputs
    

class Canny(object):

    def __call__(self, inputs):
        inputs = cv2.Canny(np.squeeze(inputs), 100, 200)
        return inputs[:,:,None]


class AddEdgeClass(object):
    def __call__(self, inputs):

        mask_float = (inputs /255).astype('float32')  
        mask_float = np.squeeze(mask_float)
        inputs = np.array(mask_float).astype('int32')

        kernel = np.ones((9,9),np.float32)/81
        mask_tmp = cv2.filter2D(mask_float,-1, kernel)   # range[0,1]
        mask_tmp = abs(mask_tmp - mask_float)
        mask_tmp = mask_tmp > 0.005
        inputs[mask_tmp] = 2

        return inputs[:,:,None].astype('int32')

class Compose(object): 

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs):
        for t in self.transforms:
            inputs = t(inputs)

        return inputs




