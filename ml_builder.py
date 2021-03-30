import numpy as np
import random as rd
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import time as tm
import os

from model.stconvs2s import STConvS2S_R, STConvS2S_C
from model.baselines import *
from model.ablation import *
 
from tool.train_evaluate import Trainer, Evaluator
from tool.dataset import NetCDFDataset
from tool.loss import RMSELoss
from tool.utils import Util
from tool.normalizer import Normalizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

class MLBuilder:

    def __init__(self, config, device):
        
        self.config = config
        self.device = device
        self.dataset_type = 'small-dataset' if (self.config.small_dataset) else 'full-dataset'
        self.x_step = config.x_step
        self.y_step = config.y_step
        self.dataset_name, self.dataset_file = self.__get_dataset_file()
        self.dropout_rate = self.__get_dropout_rate()
        self.filename_prefix = f"{self.dataset_name}_x_step_{self.x_step}_y_step_{self.y_step}"
                
    def run_model(self, number):
        self.__define_seed(number)
        validation_split = 0.2
        test_split = 0.2
        # Loading the dataset
        ds = xr.open_mfdataset(self.dataset_file)
        if (self.config.small_dataset):
            ds = ds[dict(sample=slice(0,500))]

        train_dataset = NetCDFDataset(ds, test_split=test_split, 
                                      validation_split=validation_split, x_step=self.x_step)
        val_dataset   = NetCDFDataset(ds, test_split=test_split, 
                                      validation_split=validation_split, x_step=self.x_step, is_validation=True)
        test_dataset  = NetCDFDataset(ds, test_split=test_split, 
                                      validation_split=validation_split, x_step=self.x_step, is_test=True)
        
        # normalizing data 
        num_channels = train_dataset.X.shape[1]
        if num_channels > 1:
            normalizer_x = Normalizer()
            normalizer_x.observe(train_dataset.X)
            normalizer_y = Normalizer()
            normalizer_y.observe(train_dataset.y)
            
            train_dataset.X = normalizer_x.normalize(train_dataset.X)
            train_dataset.y = normalizer_y.normalize(train_dataset.y)
            
            val_dataset.X = normalizer_x.normalize(val_dataset.X)
            val_dataset.y = normalizer_y.normalize(val_dataset.y)
            
            test_dataset.X = normalizer_x.normalize(test_dataset.X)
            test_dataset.y = normalizer_y.normalize(test_dataset.y)
            
        util = Util(self.config.model, self.dataset_type, self.config.version, self.filename_prefix)
        
        util.save_normalization_parameters(normalizer_x, normalizer_y)
        
        # INITIAL STATE - batch x channel x time x latitude x longitude
        initial_state = torch.tensor(train_dataset.X)[:1, :, :1].to(self.device)
        
        if (self.config.verbose):
            print('[X_train] Shape:', train_dataset.X.shape)
            print('[y_train] Shape:', train_dataset.y.shape)
            print('[X_val] Shape:', val_dataset.X.shape)
            print('[y_val] Shape:', val_dataset.y.shape)
            print('[X_test] Shape:', test_dataset.X.shape)
            print('[y_test] Shape:', test_dataset.y.shape)
            print(f'Train on {len(train_dataset)} samples, validate on {len(val_dataset)} samples')
                        
        params = {'batch_size': self.config.batch, 
                  'num_workers': self.config.workers, 
                  'worker_init_fn': self.__init_seed}

        train_loader = DataLoader(dataset=train_dataset, shuffle=True,**params)
        val_loader = DataLoader(dataset=val_dataset, shuffle=False,**params)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, **params)
        
        models = {
            'stconvs2s-r': STConvS2S_R,
            'stconvs2s-c': STConvS2S_C,
            'convlstm': STConvLSTM,
            'predrnn': PredRNN,
            'mim': MIM,
            'conv2plus1d': Conv2Plus1D,
            'conv3d': Conv3D,
            'enc-dec3d': Endocer_Decoder3D,
            'ablation-stconvs2s-nocausalconstraint': AblationSTConvS2S_NoCausalConstraint,
            'ablation-stconvs2s-notemporal': AblationSTConvS2S_NoTemporal,
            'ablation-stconvs2s-r-nochannelincrease': AblationSTConvS2S_R_NoChannelIncrease,
            'ablation-stconvs2s-c-nochannelincrease': AblationSTConvS2S_C_NoChannelIncrease,
            'ablation-stconvs2s-r-inverted': AblationSTConvS2S_R_Inverted,
            'ablation-stconvs2s-c-inverted': AblationSTConvS2S_C_Inverted,
            'ablation-stconvs2s-r-notfactorized': AblationSTConvS2S_R_NotFactorized,
            'ablation-stconvs2s-c-notfactorized': AblationSTConvS2S_C_NotFactorized
        }
        if not(self.config.model in models):
            raise ValueError(f'{self.config.model} is not a valid model name. Choose between: {models.keys()}')
            quit()
            
        # Creating the model    
        model_bulder = models[self.config.model]
        model = model_bulder(train_dataset.X.shape, self.config.num_layers, self.config.hidden_dim, 
                             self.config.kernel_size, self.device, self.dropout_rate, self.y_step)
        model.to(self.device)
        criterion = RMSELoss(reg=self.config.regularization, initial_state=initial_state)
        opt_params = {'lr': self.config.learning_rate, 
                      'alpha': 0.9, 
                      'eps': 1e-6}
        optimizer = torch.optim.RMSprop(model.parameters(), **opt_params)
        
        train_info = {'train_time': 0}
        if self.config.pre_trained is None:
            train_info = self.__execute_learning(model, criterion, optimizer, train_loader,  val_loader, util) 
                                                 
        eval_info = self.__load_and_evaluate(model, criterion, optimizer, test_loader, 
                                             train_info['train_time'], util)

        if (torch.cuda.is_available()):
            torch.cuda.empty_cache()

        return {**train_info, **eval_info}


    def __execute_learning(self, model, criterion, optimizer, train_loader, val_loader, util):
        checkpoint_filename = util.get_checkpoint_filename()    
        trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, self.config.epoch, 
                          self.device, util, self.config.verbose, self.config.patience, self.config.no_stop)
    
        start_timestamp = tm.time()
        # Training the model
        train_losses, val_losses = trainer.fit(checkpoint_filename, is_chirps=self.config.chirps)
        end_timestamp = tm.time()
        # Learning curve
        util.save_loss(train_losses, val_losses)
        util.plot([train_losses, val_losses], ['Training', 'Validation'], 'Epochs', 'Loss',
                  'Learning curve - ' + self.config.model.upper(), self.config.plot)

        train_time = end_timestamp - start_timestamp       
        print(f'\nTraining time: {util.to_readable_time(train_time)} [{train_time}]')
               
        return {'dataset': self.dataset_name,
                'dropout_rate': self.dropout_rate,
                'train_time': train_time
                }
                
    
    def __load_and_evaluate(self, model, criterion, optimizer, test_loader, train_time, util):  
        evaluator = Evaluator(model, criterion, optimizer, test_loader, self.device, util, self.y_step)
        if self.config.pre_trained is not None:
            # Load pre-trained model
            best_epoch, val_loss = evaluator.load_checkpoint(self.config.pre_trained, self.dataset_type, self.config.model)
        else:
            # Load model with minimal loss after training phase
            checkpoint_filename = util.get_checkpoint_filename() 
            best_epoch, val_loss = evaluator.load_checkpoint(checkpoint_filename)
        
        time_per_epochs = 0
        if not(self.config.no_stop): # Earling stopping during training
            time_per_epochs = train_time / (best_epoch + self.config.patience)
            print(f'Training time/epochs: {util.to_readable_time(time_per_epochs)} [{time_per_epochs}]')
        
        test_rmse, test_mae = evaluator.eval(is_chirps=self.config.chirps)
        print(f'Test RMSE: {test_rmse:.4f}\nTest MAE: {test_mae:.4f}')
                        
        return {'best_epoch': best_epoch,
                'val_rmse': val_loss,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'train_time_epochs': time_per_epochs
                }
          
    def __define_seed(self, number):      
        if (~self.config.no_seed):
            # define a different seed in every iteration 
            seed = (number * 10) + 1000
            np.random.seed(seed)
            rd.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic=True
            
    def __init_seed(self, number):
        seed = (number * 10) + 1000
        np.random.seed(seed)
        
    def __get_dataset_file(self):
        dataset_file, dataset_name = None, None
        if self.config.dataset:
            dataset_file = 'data/' + self.config.dataset
            dataset_name = 'shalow_water'
        else:
            if (self.config.chirps):
                dataset_file = 'data/dataset-chirps-1981-2019-seq5-ystep' + self.y_step + '.nc'
                dataset_name = 'chirps'
            else:
                dataset_file = 'data/dataset-ucar-1979-2015-seq5-ystep' + self.y_step + '.nc'
                dataset_name = 'cfsr'
        
        return dataset_name, dataset_file
        
    def __get_dropout_rate(self):
        dropout_rates = {
            'predrnn': 0.5,
            'mim': 0.5
        }
        if self.config.model in dropout_rates:
            dropout_rate = dropout_rates[self.config.model] 
        else:
            dropout_rate = 0.

        return dropout_rate