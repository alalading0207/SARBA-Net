import os
import argparse
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from lib.utils.tools.logger import Logger as Log
from lib.metrics import running_score
from lib.metrics import F1_running_score as fscore_rslib



class BaseEvaluator:

    def __init__(self, configer, trainer):
        self.configer = configer
        self.trainer = trainer
        self._init_running_scores()


    def _init_running_scores(self):
        self.running_scores = running_score.RunningScore(self.configer)
        self.metric = self.configer.get("metric")


    def update_score(self, pred, true):
        self.running_scores.update(pred, true)


    def update_performance(self):

        if self.metric == ["acc", "f1", "iou"]:
            accuracy = self.running_scores.Accuracy()
            f1_score = self.running_scores.F1_score()
            iou = self.running_scores.Iou()

            max_perf = self.configer.get('max_performance')
            self.configer.update(['performance'], f1_score)
            if f1_score > max_perf:
                Log.info('f1_score: Performance {} -> {}'.format(max_perf, f1_score))

            wandb.log({ 'metric/epoch':self.configer.get('epoch'),
                        'metric/accuracy': accuracy,
                        'metric/f1_score': f1_score,
                        'metric/iou': iou
                    })
            Log.info('Validation\tAccuracy:{:.5f}\tF1_score:{:.5f}\tIou:{:.5f}\t \n'.format(accuracy, f1_score, iou))
        

        elif self.metric == ["acc", "f1", "iou", "kappa"]:
            precision = self.running_scores.Precision()
            f1_score = self.running_scores.F1_score()
            iou = self.running_scores.Iou()
            kappa = self.running_scores.Kappa()
            boundary = self.running_scores.Boundary()


            max_perf = self.configer.get('max_performance')
            self.configer.update(['performance'], kappa)
            if kappa > max_perf:
                Log.info('kappa: Performance {} -> {}'.format(max_perf, kappa))

            wandb.log({ 'metric/epoch':self.configer.get('epoch'),
                        'metric/precision': precision,
                        'metric/f1_score': f1_score,
                        'metric/iou': iou,
                        'metric/kappa': kappa,
                        'metric/boundary': boundary
                    })
            Log.info('Validation\tPrecision:{:.5f}\tF1_score:{:.5f}\tIou:{:.5f}\tKappa:{:.5f}\tBoundary:{:.5f} \n'.format(precision, f1_score, iou, kappa, boundary))

        else:
            raise argparse.ArgumentTypeError('Unsupported metric mode...')


    def update_test(self):

        if self.metric == ["acc", "f1", "iou"]:

            accuracy = self.running_scores.Accuracy()
            f1_score = self.running_scores.F1_score()
            iou = self.running_scores.Iou()

            Log.info('Test:\t'+'Accuracy:{:.5f}\tF1_score:{:.5f}\tIou:{:.5f}\t \n'.format(accuracy, f1_score, iou))
        

        elif self.metric == ["acc", "f1", "iou", "kappa"]:
            precision = self.running_scores.Precision()
            f1_score = self.running_scores.F1_score()
            iou = self.running_scores.Iou()
            kappa = self.running_scores.Kappa()
            boundary = self.running_scores.Boundary()

            Log.info('Validation\tPrecision:{:.5f}\tF1_score:{:.5f}\tIou:{:.5f}\tKappa:{:.5f}\tBoundary:{:.5f} \n'.format(precision, f1_score, iou, kappa, boundary))

        else:
            raise argparse.ArgumentTypeError('Unsupported metric mode...')


    def reset(self):
        self.running_scores.reset()
        Log.info('The confusion matrix is reset...')


