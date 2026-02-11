import torch
import torch.nn.functional as F
from lib.utils.tools.logger import Logger as Log
import math



class CBLLoss(torch.nn.Module):
    """
    Boundary Contrast Loss.
    
    This loss enforces similarity between features of neighboring boundary pixels
    and dissimilarity to non-boundary pixels. 

    Args:
        configer: configuration handler providing hyperparameters such as
                  temperature (tau), kernel size, distance metric, etc.

    Inputs:
        features: Tensor of shape (B, C, H, W), feature maps from the backbone.
        boundary: Tensor of shape (B, 1, H, W), binary boundary map (1=boundary, 0=non-boundary).

    """

    def __init__(self, configer):
        super().__init__()
        self.configer = configer
        self.epsilon = 1e-8
        if self.configer.exists('boundary') and (self.configer.get('boundary', 'use_be') or self.configer.get('boundary', 'use_be_bc')):
            self.tau = self.configer.get('use_be', 'temperature')
            self.use_kl = self.configer.get('use_be', 'use_kl')
            self.prob = self.configer.get('use_be', 'prob')
            self.hard_prob = self.configer.get('use_be', 'hard_prob')
            self.kernel_height = self.kernel_width = self.configer.get('use_be', 'kernel_size')  # [h, w]
            Log.info('cbl_loss temperature = {}\tprob = {}\thard_prob = {}'.format(self.tau, self.prob, self.hard_prob)) 

        else:
            self.tau = 1
            self.use_kl = 3
            self.kernel_height = self.kernel_width = 3
            self.prob = 1.0
            self.hard_prob = 1.0
        self.epslion = 1e-8
        self.neibor_size = self.kernel_height * self.kernel_width - 1
        self.padding_size = int((self.kernel_height - 1)/2)
        

    def forward(self, features, boundary):
        b, c, h, w = features.shape 
        self.device = boundary.device
        loss = self.cal_dist(features, boundary.float())

        return loss


    def cal_dist(self, features, boundary):
        """
        Calculate bcl.

        Steps:
            1. Construct local neighborhoods around each pixel.
            2. Compute pairwise feature distances (KL or L2).
            3. Select valid boundary pairs.
            4. Apply contrast loss: encourage boundary pixels to be similar to nearby boundary pixels,
            and dissimilar to non-boundary pixels.
        """

        n, c, h, w = boundary.shape 

        # creat neighborhood vectors
        distance = torch.zeros([n, self.neibor_size, h, w], device=self.device)  
        label = torch.zeros([n, self.neibor_size, h, w], device=self.device)     

        # padding 
        padded_features = F.pad(features, [self.padding_size, self.padding_size, self.padding_size, self.padding_size])   
        padded_boundary = F.pad(boundary+1, [self.padding_size, self.padding_size, self.padding_size, self.padding_size], mode='constant', value=-1)


        for i in range(self.kernel_height):   
            for j in range(self.kernel_width):
                index = i*self.kernel_width+j

                if index==(self.neibor_size/2) :  
                    pass

                elif index > (self.neibor_size/2) :
                    if self.use_kl:
                        distance[:, index-1:index, :,:] = self.dist_kl(padded_features[:, :, i:i+h, j:j+w], features)
                    else:    
                        distance[:, index-1:index, :,:] = self.dist_l2(padded_features[:, :, i:i+h, j:j+w], features)  

                    label[:, index-1:index, :, :] = padded_boundary[:, :, i:i+h, j:j+w] * (boundary+1)      

                else:
                    if self.use_kl:
                        distance[:, index:index+1, :,:] = self.dist_kl(padded_features[:, :, i:i+h, j:j+w], features)
                    else:    
                        distance[:, index:index+1, :,:] = self.dist_l2(padded_features[:, :, i:i+h, j:j+w], features)  

                    label[:, index:index+1, :, :] = padded_boundary[:, :, i:i+h, j:j+w] * (boundary+1)   


        valid_point = label > 0         
        same_bou_point =  (label==4)    # i and j in the neighborhood are both boundary.
        # Number of effective boundary points: The number of central boundary points of the neighborhood that have boundary points.
        valid_bou_points =  torch.sum(torch.any(same_bou_point, dim=1))   

        distance = torch.exp(-distance / self.tau)
        # The molecule is 0 representing: 1. This point is not a boundary point. 2. This point is a boundary point, but there are no boundary points in its neighborhood.
        numerator = torch.sum(distance * same_bou_point.float(), dim=1)    
        denominator = torch.sum(distance * valid_point.float(), dim=1)     
        contrast = numerator / (denominator + self.epsilon)
        contrast[contrast==0] = 1
        every_point_log = torch.log(contrast)
        loss = (-1 / valid_bou_points) * (torch.sum(every_point_log))

        return loss
    
    
    def dist_kl(self, neighbours, features):
        log_p = features.softmax(dim=1)
        log_q = neighbours.softmax(dim=1)
        return torch.sum(log_p*(log_p.log()-log_q.log()), dim=1, keepdim=True)

    def dist_l2(self, neighbours, features):
        return torch.norm(neighbours-features, p=2, dim=1, keepdim=True) 