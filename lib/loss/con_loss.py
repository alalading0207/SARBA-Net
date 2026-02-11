import torch
import torch.nn.functional as F
import numpy as np
import cv2
import numpy as np
from PIL import Image 
from lib.utils.tools.logger import Logger as Log



def compute_grad_mag(E, device,hollow):
    E_ = convTri(E, 4, device) 
    Ox, Oy = numerical_gradients_2d(E_, device, hollow)
    mag = torch.sqrt(torch.mul(Ox, Ox) + torch.mul(Oy, Oy) + 1e-6)
    mag = mag / mag.max()

    return mag


def convTri(input, r, device):
    """
    Convolves an image by a 2D triangle filter (the 1D triangle filter f is   
    [1:r r+1 r:-1:1]/(r+1)^2, the 2D version is simply conv2(f,f'))
    :param input:
    :param r: integer filter radius  
    """
    n, c, h, w = input.shape

    # return input
    f = list(range(1, r + 1)) + [r + 1] + list(reversed(range(1, r + 1)))  # [1,2,3,4,5,4,3,2,1]
    kernel = torch.Tensor([f]) / (r + 1) ** 2  # [0.04, 0.08, 0.12, 0.16, 0.2, 0.16, 0.08, 0.04]
    kernel = kernel.to(device)

    # padding w
    input_ = F.pad(input, (1, 1, 0, 0), mode='replicate')  # pad same
    input_ = F.pad(input_, (r, r, 0, 0), mode='reflect')
    input_ = [input_[:, :, :, :r], input, input_[:, :, :, -r:]]
    input_ = torch.cat(input_, 3)
    t = input_

    # padding h
    input_ = F.pad(input_, (0, 0, 1, 1), mode='replicate')
    input_ = F.pad(input_, (0, 0, r, r), mode='reflect')
    input_ = [input_[:, :, :r, :], t, input_[:, :, -r:, :]]
    input_ = torch.cat(input_, 2)

    output = F.conv2d(input_, kernel.unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), padding=0, groups=c)
    output = F.conv2d(output, kernel.t().unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), padding=0, groups=c)

    return output


def numerical_gradients_2d(input, device, hollow):
    if hollow==True:
        x, y = gradient_central_diff_hollow(input, device)
    else:
        x, y = gradient_central_diff_solid(input, device)
    return x, y


def gradient_central_diff_solid(input_, device):  # solid: boundary + filling
    return input_, input_


def gradient_central_diff_hollow(input, device):    # hollow: only boundary
    n, c, h, w = input.shape

    # return input, input
    kernel = [[1, 0, -1]]
    kernel_t = 0.5 * torch.Tensor(kernel) * -1. 
    kernel_t = kernel_t.to(device)

    x = conv2d_same(input, kernel_t.unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), c)
    y = conv2d_same(input, kernel_t.t().unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), c)
    return x, y


def conv2d_same(input, kernel, groups,bias=None, stride=1, padding=0, dilation=1):
    n, c, h, w = input.shape  
    kout, ki_c_g, kh, kw = kernel.shape  
    pw = calc_pad_same(w, w, 1, kw)   
    ph = calc_pad_same(h, h, 1, kh)   
    pw_l = pw // 2      
    pw_r = pw - pw_l     
    ph_t = ph // 2       
    ph_b = ph - ph_t    

    result_ = F.conv2d(input, kernel, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)  
    result = F.pad(result_, (pw_l, pw_r, ph_t, ph_b), mode='replicate')  
    assert result.shape == input.shape
    return result


def calc_pad_same(in_siz, out_siz, stride, ksize):
    return (out_siz - 1) * stride + ksize - in_siz


# BCAL  
class CONLoss(torch.nn.Module):
    def __init__(self, configer):
        super(CONLoss, self).__init__()
        self.configer = configer
        self.hollow = self.configer.get('use_bc', 'hollow') if self.configer.exists("boundary") else True # use hollow
        Log.info('con_loss hollow = {}'.format(self.hollow)) 

    def forward(self, g, gts):
        device = g.device
        N, C, H, W = g.shape
        gt_th = 1e-3  
        eps = 1e-10

        g_hat = compute_grad_mag(g, device, self.hollow)
        gts_hat = compute_grad_mag(gts, device, self.hollow)

        g_hat = g_hat.view(N, -1)
        gts_hat = gts_hat.view(N, -1)
        loss_ewise = F.l1_loss(g_hat, gts_hat, reduction='none', reduce=False)

        decimal_places = 1
        g_th = torch.ceil(torch.min(g_hat) * 10**decimal_places) / 10**decimal_places
        p_plus_g_hat_mask = (g_hat >= g_th).detach().float()  
        loss_p_plus_g_hat = torch.sum(loss_ewise * p_plus_g_hat_mask) / (torch.sum(p_plus_g_hat_mask) + eps)

        p_plus_gts_hat_mask = (gts_hat >= gt_th).detach().float()  
        loss_p_plus_gts_hat = torch.sum(loss_ewise * p_plus_gts_hat_mask) / (torch.sum(p_plus_gts_hat_mask) + eps)

        total_loss = 0.5 * loss_p_plus_g_hat + 0.5 * loss_p_plus_gts_hat

        return total_loss
    


class CON_metric(torch.nn.Module):
    def __init__(self):
        super(CON_metric, self).__init__()
      
    def forward(self, g, gts):
        device = g.device
        N, C, H, W = g.shape

        g_hat = compute_grad_mag(g, device, True)
        gts_hat = compute_grad_mag(gts, device, True)

        g_hat = g_hat.view(N, -1)
        gts_hat = gts_hat.view(N, -1)
        loss_ewise = F.l1_loss(g_hat, gts_hat)

        return loss_ewise
    

class CON_out(torch.nn.Module):
    def __init__(self):
        super(CON_out, self).__init__()
      
    def forward(self, g):  # input is g or gts
        device = g.device
        N, C, H, W = g.shape

        g_hat = compute_grad_mag(g, device, True)
        
        return g_hat
    
