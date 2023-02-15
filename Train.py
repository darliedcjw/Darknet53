import torch
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision

import os
from tqdm import tqdm
from datetime import datetime

from utils import save_checkpoint
from model import Darknet53

class Train():
    
    def __init__(self,
        ds_train,
        ds_val,
        log_path,
        num_classes,
        epochs,
        batch_size,
        learning_rate,
        momentum,
        optimizer,
        num_workers,
        device,
        use_tensorboard,
        ):

        self.ds_train = ds_train
        self.ds_val = ds_val
        self.log_path = log_path
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_workers = num_workers
        self.use_tensorboard = use_tensorboard

        # Device
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        print(self.device)

        
        # SummaryWriter
        self.log_path = os.path.join(self.log_path, datetime.now().strftime('%d%m%y_%H%M%S'))
        os.makedirs(self.log_path, exist_ok=False)
        if self.use_tensorboard:
            self.summary_writer = SummaryWriter(self.log_path)

        
        # Write Parameters
        self.parameters = [x + ':' + str(y) + '\n' for x, y in locals().items()]
        with open(os.path.join(self.log_path, 'parameters.txt'), 'w') as fd:
            fd.writelines(self.parameters)
        
        if self.use_tensorboard:
            self.summary_writer.add_text('parameters', '\n'.join(self.parameters))


        # Model
        self.model = Darknet53(in_channels=3, num_classes=self.num_classes).to(self.device)

        # Optimizer
        if optimizer == 'SGD':
            self.optimizer = SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif optimizer == 'Adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer == 'RMSprop':
            self.optimizer = RMSprop(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        else:
            raise NotImplementedError('Please specify the correct optimizer!')        


        # Loss
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)


        # DataLoader
        self.dl_train = DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.len_dl_train = len(self.dl_train)
        print('Training Data Loaded: {}'.format(self.len_dl_train))

        self.dl_val = DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        self.len_dl_val = len(self.dl_val)
        print('Validation Data Loaded {}'.format(self.len_dl_val))
    
    def _train(self):

        self.model.train()

        mean_loss_train = 0
        mean_acc_train = 0

        for step, (image, label) in enumerate(tqdm(self.dl_train, desc='Training')):

            loss = 0

            image = image.to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(image)

            loss = 0.5 * self.loss_fn(output, label)

            loss.backward()

            self.optimizer.step()

            pred = torch.argmax(output, dim=1)
            acc = torch.eq(pred, label).sum() / self.batch_size

            mean_loss_train += loss.item()
            mean_acc_train += acc.item()
            
            if self.use_tensorboard: # End of a batch
                self.summary_writer.add_scalar('train_loss', loss.item(), global_step=step + self.epoch * self.len_dl_train) 
                self.summary_writer.add_scalar('train_acc', acc.item(), global_step=step + self.epoch * self.len_dl_train)


        self.mean_loss_train = mean_loss_train / self.len_dl_train
        self.mean_acc_train = mean_acc_train / self.len_dl_train

        print('Train: \nLoss: {0} \nAccuracy: {1}\n'.format(self.mean_loss_train, self.mean_acc_train))


    def _val(self):

        self.model.eval()

        mean_loss_val = 0
        mean_acc_val = 0

        with torch.no_grad():
            for step, (image, label) in enumerate(tqdm(self.dl_val, desc='Validating')):
                
                loss = 0

                image = image.to(self.device)
                label = label.to(self.device)

                output = self.model(image)

                loss = 0.5 * self.loss_fn(output, label)

                pred = torch.argmax(output, dim=1)
                acc = torch.eq(pred, label).sum() / self.batch_size

                mean_loss_val += loss.item()
                mean_acc_val += acc.item()

                if self.use_tensorboard: # End of a batch
                    self.summary_writer.add_scalar('val_loss', loss.item(), global_step=step + self.epoch * self.len_dl_val) 
                    self.summary_writer.add_scalar('val_acc', acc.item(), global_step=step + self.epoch * self.len_dl_val)
            
            self.mean_loss_val = mean_loss_val / self.len_dl_val
            self.mean_acc_val = mean_acc_val / self.len_dl_val
            
            print('Validation: \nLoss: {0} \nAccuracy: {1}\n'.format(self.mean_loss_val, self.mean_acc_val))


    def _checkpoint(self):

        save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_last.pth'), epoch=self.epoch + 1, 
                            model=self.model, optimizer=self.optimizer, params=self.parameters)
        
        if self.best_loss is None or self.best_loss > self.mean_loss_val:
            self.best_loss = self.mean_loss_val
            print('best_loss %f at epoch %d' % (self.best_loss, self.epoch + 1))
            save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_best_loss_{}.pth'.format(self.best_loss)), epoch=self.epoch + 1,
                            model=self.model, optimizer=self.optimizer, params=self.parameters)

        if self.best_acc is None or self.best_acc < self.mean_acc_val:
            self.best_acc = self.mean_acc_val
            print('best_acc %f at epoch %d' % (self.best_acc, self.epoch + 1))
            save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_best_acc_{}.pth'.format(self.best_acc)), epoch=self.epoch + 1,
                            model=self.model, optimizer=self.optimizer, params=self.parameters)

    def run(self):

        print('\nTraining started at {}'.format(datetime.now().strftime('%T-%m-%d %H:%M:%S')))

        self.best_loss = None
        self.best_acc = None

        for self.epoch in range(self.epochs):
            self.mean_loss_train = 0
            self.mean_loss_val = 0
            self.mean_acc_train = 0
            self.mean_acc_val = 0

            print('\nEpoch: {}'.format(self.epoch + 1))

            self._train()

            self._val()

            self._checkpoint()

        if self.use_tensorboard:
            self.summary_writer.close()
        
        print('\nTraining ended at {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))