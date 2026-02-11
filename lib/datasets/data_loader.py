from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import torch
from torch.utils import data
import lib.datasets.tools.transforms as trans
from lib.datasets.loader.default_loader import DefaultLoader, TestLoader
from lib.datasets.loader.large_test_loader import TestLoader_large
from lib.datasets.loader.be_loader import BELoader
from lib.datasets.loader.bc_loader import BCLoader
from lib.datasets.loader.bebc_loader import BEBCLoader
from lib.datasets.loader.ce2p_loader import CE2PLoader
from lib.datasets.loader.gscnn_loader import GSCNNLoader, GSCNN_TestLoader
from lib.datasets.loader.bfp_loader import BFPLoader
from lib.datasets.loader.decouple_loader import DecoupleLoader
from lib.utils.tools.logger import Logger as Log

class DataLoader(object):

    def __init__(self, configer):
        self.configer = configer
        self.img_transform = trans.Compose([
                                        trans.ToTensor(),
                                        trans.Normalize(div_value = self.configer.get('normalize', 'div_value'),
                                                        mean = self.configer.get('normalize', 'mean'),
                                                        std = self.configer.get('normalize', 'std'))
                                    ])
        
        self.label_transform = trans.Compose([
                                        trans.ToTensor(),
                                        trans.ToLabel(div_value = self.configer.get('normalize', 'div_value'))
                                    ]) 
        
        if self.configer.exists('other_boundary') and self.configer.get('other_boundary', 'name')=="bfp":
            self.edge_transform = trans.Compose([
                                            trans.AddEdgeClass(),
                                            trans.ToTensor()
                                        ]) 
        else:
            self.edge_transform = trans.Compose([
                                            trans.Canny(),
                                            trans.ToTensor(),
                                            trans.ToLabel(div_value = self.configer.get('normalize', 'div_value'))
                                        ]) 

    # get loader(e.g.DefaultLoader): mode/path...
    def get_dataloader(self, klass, mode):   

        root_dir = self.configer.get('data', 'data_dir')
        if isinstance(root_dir, list) and len(root_dir) == 1: 
            root_dir = root_dir[0]
        Log.info('data path:{}/{}'.format(root_dir, mode))  

        # paramater
        kwargs = dict(
            mode=mode, 
            img_transform=self.img_transform,
            label_transform=self.label_transform,
            edge_transform=self.edge_transform,
            configer=self.configer
        )
        # loader
        if isinstance(root_dir, str):            # init klass
            loader = klass(root_dir, **kwargs)   # pass the 'root_dir' and other parameters to the 'klass' --> DefaultLoader(root_dir, **kwargs)
        else:
            raise RuntimeError('Unknown root dir {}'.format(root_dir))

        return loader


    def get_trainloader(self):  # get trainloader
        
        if self.configer.exists('boundary') and self.configer.get('boundary', 'use_be'):
            Log.info('Use the BELoader for train...')
            klass = BELoader

        elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_bc') :
            Log.info('Use the BCLoader for train...')
            klass = BCLoader

        elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_be_bc') :
            Log.info('Use the BEBCLoader for train...')
            klass = BEBCLoader

        elif self.configer.exists('other_boundary') and self.configer.get('other_boundary','name')=="ce2p" :
            Log.info('Use the CE2PLoader for train...')
            klass = CE2PLoader 

        elif self.configer.exists('other_boundary') and self.configer.get('other_boundary','name')=="gscnn" :
            Log.info('Use the GSCNNLoader for train...')
            klass = GSCNNLoader 

        elif self.configer.exists('other_boundary') and self.configer.get('other_boundary','name')=="bfp" :
            Log.info('Use the BFPLoader for train...')
            klass = BFPLoader 

        elif self.configer.exists('other_boundary') and self.configer.get('other_boundary','name')=="decouple" :
            Log.info('Use the DecoupleLoader for train...')
            klass = DecoupleLoader 

        else:
            Log.info('Use the DefaultLoader for train...')
            klass = DefaultLoader    # get trainloader=DefaultLoader

        loader = self.get_dataloader(klass, 'train')   # obtain data dir and parameters
        trainloader = data.DataLoader(
            loader,  # klass  
            sampler=None,
            batch_size=self.configer.get('train', 'batch_size'), pin_memory=True,
            num_workers=self.configer.get('data', 'workers'),  shuffle=True, 
            drop_last=self.configer.get('data', 'drop_last')  
        ) 
    
        return trainloader



    def get_valloader(self, dataset=None):
        mode = 'val' if dataset is None else dataset

        if self.configer.exists('boundary') and self.configer.get('boundary', 'use_be'):
            Log.info('Use the BELoader for val...')
            klass = BELoader

        elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_bc') :
            Log.info('Use the BCLoader for val...')
            klass = BCLoader

        elif self.configer.exists('boundary') and self.configer.get('boundary', 'use_be_bc') :
            Log.info('Use the BEBCLoader for val...')
            klass = BEBCLoader
            
        elif self.configer.exists('other_boundary') and self.configer.get('other_boundary', 'name')=="ce2p" :
            Log.info('Use the CE2PLoader for train...')
            klass = CE2PLoader 

        elif self.configer.exists('other_boundary') and self.configer.get('other_boundary','name')=="gscnn" :
            Log.info('Use the GSCNNLoader for train...')
            klass = GSCNNLoader 

        elif self.configer.exists('other_boundary') and self.configer.get('other_boundary','name')=="bfp" :
            Log.info('Use the BFPLoader for train...')
            klass = BFPLoader 

        elif self.configer.exists('other_boundary') and self.configer.get('other_boundary','name')=="decouple" :
            Log.info('Use the DecoupleLoader for train...')
            klass = DecoupleLoader 
        else:
            Log.info('Use the DefaultLoader for val...')
            klass = DefaultLoader 

        loader = self.get_dataloader(klass, mode)
        valloader = data.DataLoader(
            loader,
            sampler=None,
            batch_size=self.configer.get('val', 'batch_size'), pin_memory=True,
            num_workers=self.configer.get('data', 'workers'), shuffle=False,
        )
        return valloader



    def get_testloader(self, dataset=None):
        mode = 'test' if dataset is None else dataset
        # mode = 'val' if dataset is None else dataset
        if self.configer.exists('other_boundary') and self.configer.get('other_boundary','name')=="gscnn":
            klass = GSCNN_TestLoader 
            Log.info('use GSCNN_TestLoader for Test ...')
            
        else:
            klass = TestLoader 
            Log.info('use TestLoader for Test ...')

        loader = self.get_dataloader(klass, mode)
        test_loader = data.DataLoader(
            loader,
            sampler=None,
            batch_size=self.configer.get('test', 'batch_size'), pin_memory=True,
            num_workers=self.configer.get('data', 'workers'), shuffle=False
        )
        return test_loader



    def get_testloader_large(self, dataset=None):
        mode = 'test' if dataset is None else dataset
        klass = TestLoader_large 
        Log.info('use TestLoader for Large_Test ...')

        loader = self.get_dataloader(klass, mode)
        test_loader = data.DataLoader(
            loader,
            sampler=None,
            batch_size=self.configer.get('test', 'batch_size'), pin_memory=True,
            num_workers=self.configer.get('data', 'workers'), shuffle=False
        )
        return test_loader


if __name__ == "__main__":
    pass
