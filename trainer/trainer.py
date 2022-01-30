import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from data_loader.datasets import Image_Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


#todo: figure out performance metrics

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self,
                model, 
                criterion, 
                performance_metrics, 
                optimizer,
                config, 
                device,
                dataset, 
                batch_size,
                epochs,
                transform = None,
                target_transform = None,
                num_workers = 0,
                validation_proportion = 0.2, 
                lr_scheduler=None, 
                len_epoch=None):

        super().__init__(model, criterion, performance_metrics, optimizer, config)
        self.config = config
        self.device = device
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.epochs = epochs
        if len_epoch is None:
            self.len_epoch = len(self.dataset)
        else:
            # iteration-based training
            self.data_loader = inf_loop(dataset)
            self.len_epoch = len_epoch
        self.batch_size = batch_size
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.num_workers = num_workers
        self.validation_proportion = validation_proportion
        self.log_step = int(np.sqrt(batch_size))
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)


    def train_epoch(self, 
                    train_dataloader, 
                    val_dataloader,
                    epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.performance_metrics.update('loss', loss.item())
            for met in self.performance_metrics:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        #model validation
        if self.do_validation:
            val_log = self.model_validation(val_dataloader, epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def model_validation(self, 
                        val_dataloader,
                        epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


    def train(self, dataset = None):

        if dataset == None: #train on full dataset. This method is training entry
            data, targets = self.dataset
            X_train, X_val, Y_train, Y_val = train_test_split(data,targets, test_size = self.validation_proportion)
            train_dataset = Image_Dataset(X_train,Y_train,
                                        transform = self.transform, 
                                        target_transform = self.target_transform)
            val_dataset = Image_Dataset(X_val, Y_val, 
                                        transform = self.transform, 
                                        target_transform = self.target_transform)
        else: #cross validation dataset. Method is called by cross_validation. 
            train_dataset, val_dataset = dataset
        train_dataloader = DataLoader(train_dataset,
                                    batch_size = self.batch_size, 
                                    shuffle=True, 
                                    num_workers = self.num_workers, 
                                    transform = self.transform)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size = self.batch_size, 
                                    shuffle=True, training=False,
                                    num_workers = self.num_workers, 
                                    target_transform = self.target_transform)
        for epoch in range(self.epochs):
            #TODO: fgure out logging
            training_log =  self.train_epoch(train_dataloader, val_dataloader, epoch)


    def cross_validation(self, folds = 5):
        data, targets = self.dataset
        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        fold = 0
        for train_idx, val_idx in skf.split(data, targets):
            X_train, X_val = data[train_idx], data[val_idx]
            Y_train, Y_val = targets[train_idx], targets[val_idx]
            train_dataset = Image_Dataset(X_train,Y_train, 
                                        transform = self.transform, 
                                        target_transform = self.target_transform)
            val_dataset = Image_Dataset(X_val, Y_val, 
                                        transform = self.transform, 
                                        target_transform = self.target_transform)
            dataset = train_dataset, val_dataset
            return_val = self.train(dataset)
            fold +=1 
  
