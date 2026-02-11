from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import argparse
import wandb
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.tools.average_meter import AverageMeter
from lib.utils.tools.logger import Logger as Log
from segmentor.tools.data_helper import DataHelper
from segmentor.tools.evaluator import get_evaluator
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler


class Trainer(object):
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.foward_time = AverageMeter()
        self.backward_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = {'pixel':AverageMeter(), 'cbl':AverageMeter(), 'bce':AverageMeter(), 'con':AverageMeter(), 'aux':AverageMeter(), 'edge':AverageMeter()}  # pixel, cbl, bce, con
        self.val_losses = {'pixel':AverageMeter(), 'cbl':AverageMeter(), 'bce':AverageMeter(), 'con':AverageMeter(), 'aux':AverageMeter(), 'edge':AverageMeter()}   # pixel, cbl, bce, con
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.data_loader = DataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.data_helper = DataHelper(configer, self)
        self.evaluator = get_evaluator(configer, self)

        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.running_score = None
        self.current_lr = []

        self._init_model()

    def _init_model(self):
        
        # net
        self.seg_net = self.model_manager.semantic_segmentor()  
        self.seg_net = self.module_runner.load_net(self.seg_net)  

        # optimizer
        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(self.seg_net.parameters())

        # dataset
        self.train_loader = self.data_loader.get_trainloader()
        self.val_loader = self.data_loader.get_valloader()

        # loss
        self.pixel_loss = self.loss_manager.get_seg_loss(self.configer.get("loss", "loss_type")) 
        self.cbl_loss = self.loss_manager.get_seg_loss('cbl_loss')
        self.bce_loss = self.loss_manager.get_seg_loss('bce_loss')
        self.con_loss = self.loss_manager.get_seg_loss('con_loss')
        self.ce_loss = self.loss_manager.get_seg_loss('ce_loss')


    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        self.seg_net.train()
        self.pixel_loss.train()
        self.cbl_loss.train()
        self.bce_loss.train()
        self.con_loss.train()
        self.ce_loss.train()
        start_time = time.time()

        for i, data_dict in enumerate(self.train_loader):

            if self.configer.get('lr', 'metric') == 'iters':
                self.scheduler.step(self.configer.get('iters'))   
            else: 
                self.scheduler.step(self.configer.get('epoch'))  


            if not self.configer.exists("boundary") and not self.configer.exists("other_boundary"):
                (img, labelmap), batch_size= self.data_helper.prepare_data(data_dict)
                self.data_time.update(time.time() - start_time)
                
                foward_start_time = time.time()
                outputs = self.seg_net(img)
                self.foward_time.update(time.time() - foward_start_time)

                loss_start_time = time.time()
                backward_loss = display_loss = pixel_loss = self.pixel_loss(outputs, labelmap)
                self.train_losses['pixel'].update(pixel_loss.item(), batch_size)
                self.loss_time.update(time.time() - loss_start_time)


            elif self.configer.exists('other_boundary') and self.configer.get('other_boundary', 'name')=="ce2p":   
                (img, labelmap, boundary), batch_size = self.data_helper.prepare_data(data_dict)
                self.data_time.update(time.time() - start_time)
                
                foward_start_time = time.time()
                output_aux, output, edge = self.seg_net(img)
                self.foward_time.update(time.time() - foward_start_time)

                loss_start_time = time.time()
                pixel_loss = self.pixel_loss(output, labelmap)
                aux_loss = self.pixel_loss(output_aux, labelmap)
                edge_loss = self.bce_loss(edge, boundary)
                backward_loss = display_loss = pixel_loss + \
                                self.configer.get('other_boundary', 'aux_weight') * aux_loss + \
                                self.configer.get('other_boundary', 'edge_weight') * edge_loss 
                self.train_losses['pixel'].update(pixel_loss.item(), batch_size)
                self.train_losses['aux'].update(aux_loss.item(), batch_size)
                self.train_losses['edge'].update(edge_loss.item(), batch_size)
                self.loss_time.update(time.time() - loss_start_time)

            elif self.configer.exists('other_boundary') and self.configer.get('other_boundary', 'name')=="decouple":   
                (img, labelmap, boundary), batch_size = self.data_helper.prepare_data(data_dict)
                self.data_time.update(time.time() - start_time)
                
                foward_start_time = time.time()
                output, output_aux, edge = self.seg_net(img)
                self.foward_time.update(time.time() - foward_start_time)

                loss_start_time = time.time()
                pixel_loss = self.pixel_loss(output, labelmap)
                aux_loss = self.pixel_loss(output_aux, labelmap)
                edge_loss = self.bce_loss(edge, boundary)
                backward_loss = display_loss = pixel_loss + \
                                self.configer.get('other_boundary', 'aux_weight') * aux_loss + \
                                self.configer.get('other_boundary', 'edge_weight') * edge_loss 
                self.train_losses['pixel'].update(pixel_loss.item(), batch_size)
                self.train_losses['aux'].update(aux_loss.item(), batch_size)
                self.train_losses['edge'].update(edge_loss.item(), batch_size)
                self.loss_time.update(time.time() - loss_start_time)

            elif self.configer.exists('other_boundary') and self.configer.get('other_boundary', 'name')=="gscnn":   
                (img, canny, labelmap, boundary), batch_size = self.data_helper.prepare_data(data_dict)
                self.data_time.update(time.time() - start_time)
                
                foward_start_time = time.time()
                output, con, edge = self.seg_net(img,canny)
                self.foward_time.update(time.time() - foward_start_time)

                loss_start_time = time.time()
                pixel_loss = self.pixel_loss(output, labelmap)
                edge_loss = self.bce_loss(edge, boundary)
                aux_loss = self.con_loss(con, labelmap)
                backward_loss = display_loss = pixel_loss + \
                                self.configer.get('other_boundary', 'edge_weight') * edge_loss + \
                                self.configer.get('other_boundary', 'aux_weight') * aux_loss 
                self.train_losses['pixel'].update(pixel_loss.item(), batch_size)
                self.train_losses['aux'].update(aux_loss.item(), batch_size)
                self.train_losses['edge'].update(edge_loss.item(), batch_size)
                self.loss_time.update(time.time() - loss_start_time)

            elif self.configer.exists('other_boundary') and self.configer.get('other_boundary', 'name')=="bfp":   
                (img, labelmap, label_boundary), batch_size = self.data_helper.prepare_data(data_dict)
                self.data_time.update(time.time() - start_time)
                
                foward_start_time = time.time()
                output, out_edge = self.seg_net(img)
                self.foward_time.update(time.time() - foward_start_time)

                loss_start_time = time.time()
                pixel_loss = self.pixel_loss(output, labelmap)
                edge_loss = self.ce_loss(out_edge, label_boundary.squeeze(dim=1))
                backward_loss = display_loss = pixel_loss + \
                                self.configer.get('other_boundary', 'edge_weight') * edge_loss
                self.train_losses['pixel'].update(pixel_loss.item(), batch_size)
                self.train_losses['edge'].update(edge_loss.item(), batch_size)
                self.loss_time.update(time.time() - loss_start_time)


            elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_be'):
                (img, labelmap, sub_boundary), batch_size = self.data_helper.prepare_data(data_dict)
                self.data_time.update(time.time() - start_time)
                
                foward_start_time = time.time()
                output, cbl, bce = self.seg_net(img)
                self.foward_time.update(time.time() - foward_start_time)

                loss_start_time = time.time()
                pixel_loss = self.pixel_loss(output, labelmap)  
                cbl_loss, bce_loss = 0, 0
                resolution = self.configer.get('boundary', 'resolution')   # "use_be": [2,4,8]   [1,2,4,8]   [1,4,8] 
                for i, r in enumerate(resolution):
                    cbl_loss += self.cbl_loss(cbl['cbl_1_{}'.format(r)], sub_boundary[i])
                    bce_loss += self.bce_loss(bce['bce_1_{}'.format(r)], sub_boundary[i])
                backward_loss = display_loss = pixel_loss + \
                                self.configer.get('boundary', 'cbl_weight') * cbl_loss + \
                                self.configer.get('boundary', 'bce_weight') * bce_loss
                self.train_losses['pixel'].update(pixel_loss.item(), batch_size)
                self.train_losses['cbl'].update(cbl_loss.item(), batch_size)
                self.train_losses['bce'].update(bce_loss.item(), batch_size)
                self.loss_time.update(time.time() - loss_start_time)


            elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_bc'):
                (img, labelmap, sub_label), batch_size= self.data_helper.prepare_data(data_dict)
                self.data_time.update(time.time() - start_time)
                
                foward_start_time = time.time()
                output, con = self.seg_net(img)
                self.foward_time.update(time.time() - foward_start_time)

                loss_start_time = time.time()
                pixel_loss = self.pixel_loss(output, labelmap)  
                con_loss = 0
                resolution = self.configer.get('boundary', 'resolution')   # "use_be": [2,4,8]   [1,2,4,8]   [1,4,8] 
                for i, r in enumerate(resolution):
                    con_loss += self.con_loss(con['con_1_{}'.format(r)], sub_label[i])
                backward_loss = display_loss = pixel_loss + \
                                self.configer.get('boundary', 'con_weight') * con_loss 
                self.train_losses['pixel'].update(pixel_loss.item(), batch_size)
                self.train_losses['con'].update(con_loss.item(), batch_size)
                self.loss_time.update(time.time() - loss_start_time)


            elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_be_bc'):
                (img, labelmap, sub_boundary, labelmap2), batch_size= self.data_helper.prepare_data(data_dict)
                self.data_time.update(time.time() - start_time)
                
                foward_start_time = time.time()
                output, cbl, bce, con = self.seg_net(img)
                self.foward_time.update(time.time() - foward_start_time)

                loss_start_time = time.time()
                pixel_loss = self.pixel_loss(output, labelmap) 
                cbl_loss, bce_loss, con_loss = 0, 0, 0
                resolution = self.configer.get('boundary', 'resolution')   # "use_be": [2,4,8]   [1,2,4,8]   [1,4,8] 
                for i, r in enumerate(resolution):
                    cbl_loss += self.cbl_loss(cbl['cbl_1_{}'.format(r)], sub_boundary[i])
                    bce_loss += self.bce_loss(bce['bce_1_{}'.format(r)], sub_boundary[i])
                con_loss = self.con_loss(con, labelmap2)
                backward_loss = display_loss = pixel_loss + \
                                self.configer.get('boundary', 'cbl_weight') * cbl_loss + \
                                self.configer.get('boundary', 'bce_weight') * bce_loss + \
                                self.configer.get('boundary', 'con_weight') * con_loss
                self.train_losses['pixel'].update(pixel_loss.item(), batch_size)
                self.train_losses['cbl'].update(cbl_loss.item(), batch_size)
                self.train_losses['bce'].update(bce_loss.item(), batch_size)
                self.train_losses['con'].update(con_loss.item(), batch_size)
                self.loss_time.update(time.time() - loss_start_time)

            else:
                raise argparse.ArgumentTypeError('Unsupported training mode...')


            backward_start_time = time.time()
            self.optimizer.zero_grad()
            backward_loss.backward()
            self.optimizer.step()
            self.backward_time.update(time.time() - backward_start_time)


            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one('iters')


            # Print the log info & reset the states.
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0:
                Log.info('Train Epoch: {0}\tTrain Iteration: {1}\tLearning rate = {3}\t'
                         'Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t'
                         'Forward Time {foward_time.sum:.3f}s / {2}iters, ({foward_time.avg:.3f})\t'
                         'Backward Time {backward_time.sum:.3f}s / {2}iters, ({backward_time.avg:.3f})\t'
                         'Loss Time {loss_time.sum:.3f}s / {2}iters, ({loss_time.avg:.3f})\t'
                         'Data Time {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                         'Display Loss(current) = {display_loss:.5f} \t Pixel Loss = {pixel_loss:.5f} \n'.format(
                    self.configer.get('epoch'), self.configer.get('iters'),
                    self.configer.get('solver', 'display_iter'),
                    self.module_runner.get_lr(self.optimizer), batch_time=self.batch_time,
                    foward_time=self.foward_time, backward_time=self.backward_time, loss_time=self.loss_time,
                    data_time=self.data_time, display_loss=display_loss, pixel_loss = pixel_loss))
                                
                if not self.configer.exists('other_boundary'):
                    wandb.log({
                            "train/epoch": self.configer.get('epoch'),
                            'train/lr': self.scheduler.get_lr()[0],
                            'train/loss_pixel': self.train_losses['pixel'].avg,
                            'train/loss_cbl': self.train_losses['cbl'].avg,
                            'train/loss_bce': self.train_losses['bce'].avg,
                            'train/loss_con': self.train_losses['con'].avg
                        })
                
                else:
                    wandb.log({
                            "train/epoch": self.configer.get('epoch'),
                            'train/lr': self.scheduler.get_lr()[0],
                            'train/loss_pixel': self.train_losses['pixel'].avg,
                            'train/loss_aux': self.train_losses['aux'].avg,
                            'train/loss_edge': self.train_losses['edge'].avg
                        })
                                    
                self.batch_time.reset()
                self.foward_time.reset()
                self.backward_time.reset()
                self.loss_time.reset()
                self.data_time.reset()
                self.train_losses['pixel'].reset()
                self.train_losses['cbl'].reset()
                self.train_losses['bce'].reset()
                self.train_losses['con'].reset()
                self.train_losses['aux'].reset()
                self.train_losses['edge'].reset()


        Log.info('##################### One epoch end ####################\n')
        self.configer.plus_one('epoch')
        self.configer.update(['iters'], 0)
        self.__val()



    def __val(self, data_loader=None):
        """
          Validation function during the train phase.
        """
        Log.info('##################### Start Validation ####################')
        self.seg_net.eval()
        self.pixel_loss.eval()
        self.cbl_loss.eval()
        self.bce_loss.eval()
        self.con_loss.eval()
        self.ce_loss.eval()
        start_time = time.time()

        data_loader = self.val_loader if data_loader is None else data_loader
        for j, data_dict in enumerate(data_loader):

            if not self.configer.exists("boundary") and not self.configer.exists("other_boundary"):
                (img, labelmap), batch_size= self.data_helper.prepare_data(data_dict)
                with torch.no_grad():
                    output = self.seg_net(img)
                    loss = pixel_loss = self.pixel_loss(output, labelmap)
                    self.val_losses['pixel'].update(pixel_loss.item(), batch_size)


            elif self.configer.exists('other_boundary') and self.configer.get('other_boundary', 'name')=="ce2p":   
                (img, labelmap, boundary), batch_size= self.data_helper.prepare_data(data_dict)
                with torch.no_grad():
                    output_aux, output, edge = self.seg_net(img)
                    pixel_loss = self.pixel_loss(output, labelmap)
                    aux_loss = self.pixel_loss(output_aux, labelmap)
                    edge_loss = self.bce_loss(edge, boundary)
                    loss = pixel_loss + \
                           self.configer.get('other_boundary', 'aux_weight') * aux_loss + \
                           self.configer.get('other_boundary', 'edge_weight') * edge_loss 
                self.val_losses['pixel'].update(pixel_loss.item(), batch_size)
                self.val_losses['aux'].update(aux_loss.item(), batch_size)
                self.val_losses['edge'].update(edge_loss.item(), batch_size)

            elif self.configer.exists('other_boundary') and self.configer.get('other_boundary', 'name')=="decouple":   
                (img, labelmap, boundary), batch_size= self.data_helper.prepare_data(data_dict)
                with torch.no_grad():
                    output, output_aux, edge = self.seg_net(img)
                    pixel_loss = self.pixel_loss(output, labelmap)
                    aux_loss = self.pixel_loss(output_aux, labelmap)
                    edge_loss = self.bce_loss(edge, boundary)
                    loss = pixel_loss + \
                           self.configer.get('other_boundary', 'aux_weight') * aux_loss + \
                           self.configer.get('other_boundary', 'edge_weight') * edge_loss 
                self.val_losses['pixel'].update(pixel_loss.item(), batch_size)
                self.val_losses['aux'].update(aux_loss.item(), batch_size)
                self.val_losses['edge'].update(edge_loss.item(), batch_size)


            elif self.configer.exists('other_boundary') and self.configer.get('other_boundary', 'name')=="gscnn":   
                (img, canny, labelmap, boundary), batch_size= self.data_helper.prepare_data(data_dict)
                with torch.no_grad():
                    output, con, edge = self.seg_net(img, canny)
                    pixel_loss = self.pixel_loss(output, labelmap)
                    edge_loss = self.bce_loss(edge, boundary)
                    aux_loss = self.con_loss(con, labelmap)
                    loss = pixel_loss + \
                           self.configer.get('other_boundary', 'edge_weight') * edge_loss + \
                           self.configer.get('other_boundary', 'aux_weight') * aux_loss
                self.val_losses['pixel'].update(pixel_loss.item(), batch_size)
                self.val_losses['aux'].update(aux_loss.item(), batch_size)
                self.val_losses['edge'].update(edge_loss.item(), batch_size)

            elif self.configer.exists('other_boundary') and self.configer.get('other_boundary', 'name')=="bfp":   
                (img, labelmap, label_boundary), batch_size= self.data_helper.prepare_data(data_dict)
                with torch.no_grad():
                    output, out_edge = self.seg_net(img)
                    pixel_loss = self.pixel_loss(output, labelmap)
                    edge_loss = self.ce_loss(out_edge, label_boundary.squeeze(dim=1))
                    loss = pixel_loss + \
                           self.configer.get('other_boundary', 'edge_weight') * edge_loss
                self.val_losses['pixel'].update(pixel_loss.item(), batch_size)
                self.val_losses['edge'].update(edge_loss.item(), batch_size)


            elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_be'):
                (img, labelmap, sub_boundary), batch_size= self.data_helper.prepare_data(data_dict)
                with torch.no_grad():
                    output, cbl, bce = self.seg_net(img)
                    pixel_loss = self.pixel_loss(output, labelmap)  
                    cbl_loss, bce_loss = 0, 0
                    resolution = self.configer.get('boundary', 'resolution')   # "use_be": [2,4,8]   [1,2,4,8]   [1,4,8] 
                    for i, r in enumerate(resolution):
                        cbl_loss += self.cbl_loss(cbl['cbl_1_{}'.format(r)], sub_boundary[i])
                        bce_loss += self.bce_loss(bce['bce_1_{}'.format(r)], sub_boundary[i])
                    loss = pixel_loss + \
                            self.configer.get('boundary', 'cbl_weight') * cbl_loss + \
                            self.configer.get('boundary', 'bce_weight') * bce_loss
                    self.val_losses['pixel'].update(pixel_loss.item(), batch_size)
                    self.val_losses['cbl'].update(cbl_loss.item(), batch_size)
                    self.val_losses['bce'].update(bce_loss.item(), batch_size)


            elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_bc'):
                (img, labelmap, sub_label), batch_size= self.data_helper.prepare_data(data_dict)
                with torch.no_grad():
                    output, con = self.seg_net(img)
                    pixel_loss = self.pixel_loss(output, labelmap)  
                    con_loss = 0
                    resolution = self.configer.get('boundary', 'resolution')   # "use_be": [2,4,8]   [1,2,4,8]   [1,4,8] 
                    for i, r in enumerate(resolution):
                        if  r == 1:
                            con_loss += self.con_loss(con['con_1_{}'.format(r)], labelmap)
                        else:
                            con_loss += self.con_loss(con['con_1_{}'.format(r)], sub_label[i])
                    loss = pixel_loss + \
                            self.configer.get('boundary', 'con_weight') * con_loss 
                    self.val_losses['pixel'].update(pixel_loss.item(), batch_size)
                    self.val_losses['con'].update(con_loss.item(), batch_size)


            elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_be_bc'):
                (img, labelmap, sub_boundary, labelmap2), batch_size= self.data_helper.prepare_data(data_dict)
                with torch.no_grad():
                    output, cbl, bce, con = self.seg_net(img)
                    pixel_loss = self.pixel_loss(output, labelmap)  
                    cbl_loss, bce_loss, con_loss = 0, 0, 0
                    resolution = self.configer.get('boundary', 'resolution')   # "use_be": [2,4,8]   [1,2,4,8]   [1,4,8] 
                    for i, r in enumerate(resolution):
                        cbl_loss += self.cbl_loss(cbl['cbl_1_{}'.format(r)], sub_boundary[i])
                        bce_loss += self.bce_loss(bce['bce_1_{}'.format(r)], sub_boundary[i])

                    con_loss = self.con_loss(con, labelmap2)
                    loss = pixel_loss + \
                            self.configer.get('boundary', 'cbl_weight') * cbl_loss + \
                            self.configer.get('boundary', 'bce_weight') * bce_loss + \
                            self.configer.get('boundary', 'con_weight') * con_loss 
                    self.val_losses['pixel'].update(pixel_loss.item(), batch_size)
                    self.val_losses['cbl'].update(cbl_loss.item(), batch_size)
                    self.val_losses['bce'].update(bce_loss.item(), batch_size)
                    self.val_losses['con'].update(con_loss.item(), batch_size)
                    
            self.evaluator.update_score(output, labelmap)
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()


        if not self.configer.exists('other_boundary'):
            wandb.log({
                        'val/epoch': self.configer.get('epoch'),
                        'val/loss_pixel': self.val_losses['pixel'].avg,
                        'val/loss_cbl': self.val_losses['cbl'].avg,
                        'val/loss_bce': self.val_losses['bce'].avg,
                        'val/loss_con': self.val_losses['con'].avg
                    })
        else:
            wandb.log({
                        'val/epoch': self.configer.get('epoch'),
                        'val/loss_pixel': self.val_losses['pixel'].avg,
                        'train/loss_aux': self.val_losses['aux'].avg,
                        'train/loss_edge': self.val_losses['edge'].avg
                        })
            

        self.configer.update(['val_loss'], self.val_losses['pixel'].avg)
        self.evaluator.update_performance()

        self.module_runner.save_net(self.seg_net, self.optimizer, self.scheduler, save_mode='performance', experiment=None)
        self.module_runner.save_net(self.seg_net, self.optimizer, self.scheduler, save_mode='val_loss', experiment=None)
        cudnn.benchmark = True

        Log.info('Validation Time {batch_time.sum:.3f}s, (ave:{batch_time.avg:.3f})\t'
                 'Total loss(current) = {Total_loss:.5f}\tPixel loss = {Pixel_loss:.5f}\n'
                 .format(batch_time=self.batch_time, Total_loss = loss, Pixel_loss = pixel_loss))

        self.batch_time.reset()
        self.evaluator.reset()
        self.val_losses['pixel'].reset()
        self.val_losses['cbl'].reset()
        self.val_losses['bce'].reset()
        self.val_losses['con'].reset()
        self.val_losses['aux'].reset()
        self.val_losses['edge'].reset()
        self.seg_net.train()
        self.pixel_loss.train()
        self.cbl_loss.train()
        self.bce_loss.train()
        self.con_loss.train()
        Log.info('##################### Validation End ####################\n')



    def train(self):

        while self.configer.get('epoch') < self.configer.get('solver', 'max_epoch'):
            self.__train()

        self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))


if __name__ == "__main__":
    pass
