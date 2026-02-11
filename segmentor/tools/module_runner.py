from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from lib.utils.tools.logger import Logger as Log


class ModuleRunner(object):

    def __init__(self, configer):
        self.configer = configer
        self._init()

    def _init(self):
        self.configer.add(['iters'], 0)
        self.configer.add(['last_iters'], 0)
        self.configer.add(['epoch'], 0)
        self.configer.add(['last_epoch'], 0)
        self.configer.add(['max_performance'], 0.0) 
        self.configer.add(['performance'], 0.0)
        self.configer.add(['min_val_loss'], 9999.0)
        self.configer.add(['val_loss'], 9999.0)
        if not self.configer.exists('network', 'bn_type'):
            self.configer.add(['network', 'bn_type'], 'torchbn')
        # Log.info('BN Type is {}.'.format(self.configer.get('network', 'bn_type')))



    def to_device_net(self, *params, force_list=False): 
        device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        return_list = list()
        for i in range(len(params)):   # e.g. params -> hrnet  
            return_list.append(params[i].to(device)) 

        if force_list:
            return return_list   
        else:
            return return_list[0] if len(params) == 1 else return_list   # return_list[0] only one params


    def to_device_data(self, data): 
        device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        if isinstance(data, torch.Tensor):
            return data.to(device)

        elif isinstance(data, list):
            return_list = list()
            for i in range(len(data)): 
                return_list.append(data[i].to(device)) 
            # return return_list[0] if len(data) == 1 else return_list
            return return_list
        else:
            raise argparse.ArgumentTypeError('Data Type need torch.Tensor or list...')


    # net to GPU
    def load_net(self, net):    
        net = self.to_device_net(net)
        net.float()

        # resume
        if self.configer.get('network', 'resume') is not None:  
            Log.info('Loading checkpoint from {}'.format(self.configer.get('network', 'resume')))  
            resume_dict = torch.load(self.configer.get('network', 'resume'), map_location=lambda storage, loc: storage) 
            
            if 'state_dict' in resume_dict:
                checkpoint_dict = resume_dict['state_dict'] # √

            elif 'model' in resume_dict:
                checkpoint_dict = resume_dict['model']

            elif isinstance(resume_dict, OrderedDict):
                checkpoint_dict = resume_dict

            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(self.configer.get('network', 'resume')))

            if list(checkpoint_dict.keys())[0].startswith('module.'):  
                checkpoint_dict = {k[7:]: v for k, v in checkpoint_dict.items()}

            # load state_dict
            if hasattr(net, 'module'):
                self.load_state_dict(net.module, checkpoint_dict, self.configer.get('network', 'resume_strict'))
            else:
                self.load_state_dict(net, checkpoint_dict, self.configer.get('network', 'resume_strict'))  

            # load config_dict
            if self.configer.get('network', 'resume_continue'):
                self.configer.resume(resume_dict['config_dict'])
                print(resume_dict['config_dict'])
                # self.configer.update(['network', 'resume'], None)

        return net


    @staticmethod
    def load_state_dict(module, state_dict, strict=False):  
        """Load state_dict to a module.
        This method is modified from :meth:`torch.nn.Module.load_state_dict`.
        Default value for ``strict`` is set to ``False`` and the message for
        param mismatch will be shown even if strict is False.
        Args:
            module (Module): Module that receives the state_dict.
            state_dict (OrderedDict): Weights.
            strict (bool): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        """
        unexpected_keys = []
        own_state = module.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                unexpected_keys.append(name)
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            try:
                own_state[name].copy_(param)
            except Exception:
                Log.warn('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(),
                                           param.size()))
                
        missing_keys = set(own_state.keys()) - set(state_dict.keys())

        err_msg = []
        if unexpected_keys:
            err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
        if missing_keys:
            # we comment this to fine-tune the models with some missing keys.
            err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
        err_msg = '\n'.join(err_msg)
        if err_msg:
            if strict:
                raise RuntimeError(err_msg)
            else:
                Log.warn(err_msg)



    def save_net(self, net, optimizer, scheduler, save_mode='iters', experiment=None):   
        state = {
            'config_dict': self.configer.to_dict(),
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if self.configer.get('checkpoints', 'checkpoints_root') is None:
            checkpoints_dir = os.path.join(self.configer.get('project_dir'),
                                           self.configer.get('checkpoints', 'checkpoints_dir'))
        else:  
            checkpoints_dir = os.path.join(self.configer.get('checkpoints', 'checkpoints_root'),  
                                           self.configer.get('checkpoints', 'checkpoints_dir'))   

        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        latest_name = '{}_latest.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'))
        torch.save(state, os.path.join(checkpoints_dir, latest_name)) 
        
        if save_mode == 'performance':   # max_performance
            if self.configer.get('performance') > self.configer.get('max_performance'):
                latest_name = '{}_max_performance.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update(['max_performance'], self.configer.get('performance'))

        elif save_mode == 'val_loss':   # min_val_loss
            if self.configer.get('val_loss') < self.configer.get('min_val_loss'):
                latest_name = '{}_min_loss.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update(['min_val_loss'], self.configer.get('val_loss'))

        elif save_mode == 'iters':
            if self.configer.get('iters') - self.configer.get('last_iters') >= \
                    self.configer.get('checkpoints', 'save_iters'):
                latest_name = '{}_iters{}.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'),
                                                 self.configer.get('iters'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update(['last_iters'], self.configer.get('iters'))

        elif save_mode == 'epoch':
            if self.configer.get('epoch') - self.configer.get('last_epoch') >= \
                    self.configer.get('checkpoints', 'save_epoch'):
                latest_name = '{}_epoch{}.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'),
                                                 self.configer.get('epoch'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update(['last_epoch'], self.configer.get('epoch'))

        else:
            Log.error('Metric: {} is invalid.'.format(save_mode))
            exit(1)

        if experiment is not None:
            experiment.checkpoint(
                path=os.path.join(checkpoints_dir, latest_name),
                step=self.configer.get('iters'),
                metrics={'mIoU': self.configer.get('performance'), 'loss': self.configer.get('val_loss')},
                primary_metric=("mIoU", "maximize")
            )


    def freeze_bn(self, net, syncbn=False):   
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()

            if syncbn:
                from lib.extensions import BatchNorm2d, BatchNorm1d
                if isinstance(m, BatchNorm2d) or isinstance(m, BatchNorm1d):
                    m.eval()

    def clip_grad(self, model, max_grad=10.):   
        """Computes a gradient clipping coefficient based on gradient norm."""
        total_norm = 0
        for p in model.parameters():
            if p.requires_grad:
                modulenorm = p.grad.data.norm()
                total_norm += modulenorm ** 2

        total_norm = math.sqrt(total_norm)

        norm = max_grad / max(total_norm, max_grad)
        for p in model.parameters():
            if p.requires_grad:
                p.grad.mul_(norm)



    def get_lr(self, optimizer):    
        return [param_group['lr'] for param_group in optimizer.param_groups]



    def warm_lr(self, iters, scheduler, optimizer, backbone_list=(0, )): 
        if not self.configer.exists('lr', 'is_warm') or not self.configer.get('lr', 'is_warm'):
            return

        warm_iters = self.configer.get('lr', 'warm')['warm_iters']
        if iters < warm_iters:
            if self.configer.get('lr', 'warm')['freeze_backbone']:
                for backbone_index in backbone_list:
                    optimizer.param_groups[backbone_index]['lr'] = 0.0

            else:
                lr_ratio = (self.configer.get('iters') + 1) / warm_iters
                base_lr_list = scheduler.get_lr()
                for backbone_index in backbone_list:
                    optimizer.param_groups[backbone_index]['lr'] = base_lr_list[backbone_index] * (lr_ratio ** 4)

