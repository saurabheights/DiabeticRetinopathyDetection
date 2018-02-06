from copy import deepcopy
from itertools import chain
from pathlib import Path

import progressbar
import torch
from torch.autograd import Variable
import time
import logging

from quadratic_weighted_kappa import quadratic_weighted_kappa


class ModelTrainer:
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, model, train_dataset_loader, valid_dataset_loader, test_dataset_loader,
                 model_path,
                 scheduler,
                 host_device='cpu',
                 optimizer=torch.optim.Adam,
                 optimizer_args={},                 
                 loss_func=torch.nn.CrossEntropyLoss(size_average=False),
                 patience=float('Inf')):
        self.model = model
        self.train_dataset_loader = train_dataset_loader
        self.valid_dataset_loader = valid_dataset_loader
        self.test_dataset_loader = test_dataset_loader
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(exist_ok=True)

        self.host_device = host_device
        self.optimizer_args = optimizer_args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func

        self._reset_histories()
        self.patience = patience
        self.wait = 0
        self.best_qwk = -1
        self.best_model = None

    def _reset_histories(self):
        """
        Resets train and val histories for the qwkuracy and the loss.
        """
        self.train_loss_history = []
        self.train_qwk_history = []
        self.val_loss_history = []
        self.val_qwk_history = []

    def train_model(self, num_epochs, log_nth):
        training_start_time = time.time()

        optimizer = self.optimizer
        
        self._reset_histories()
        if self.host_device == 'gpu':
            self.model.cuda()
        iter_per_epoch = len(self.train_dataset_loader)
        logging.info("Start training")
        logging.info(f"Size of training data: "
                     f"{len(self.train_dataset_loader.sampler) * self.train_dataset_loader.batch_size}")
        
        
        for i_epoch in range(num_epochs):
            logging.info("Starting new epoch...")
            running_loss = 0.
            
            
            all_y = []
            all_y_pred = []            
            
            # scheduler step for exp and step schedulers
            
            if (not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                self.scheduler.step()
                logging.info(f"Learning rate is {self.scheduler.get_lr()}")
                
            for i_batch, batch in enumerate(self.train_dataset_loader):
                x, y = batch
                x, y = Variable(x), Variable(y)
                if self.host_device == 'gpu':
                    x, y = x.cuda(), y.cuda()

                optimizer.zero_grad()
                outputs = self.model(x)
                if self.host_device == 'gpu':
                    train_loss = self.loss_func(outputs.cuda(), y)
                else:
                    train_loss = self.loss_func(outputs, y)

                train_loss.backward()
                optimizer.step()

                running_loss += train_loss.data[0]
                _, y_pred = torch.max(outputs.data, 1)
                all_y.append(y)
                all_y_pred.append(y_pred)

                if not log_nth == 0 and (i_batch % log_nth) == 0:
                    logging.info(f'[Iteration {i_batch}/{iter_per_epoch}] '
                          f'TRAIN loss: {running_loss / sum(curr_y.shape[0] for curr_y in all_y):.3f}')
                self.train_loss_history.append(running_loss)
            y = torch.cat(all_y)
            y_pred = torch.cat(all_y_pred)
            train_qwk = quadratic_weighted_kappa(y_pred, y.data)

            logging.info(f'[Epoch {i_epoch+1}/{num_epochs}] '
                  f'TRAIN   QWK: {train_qwk:.3f}; loss: {running_loss / y.shape[0]:.3f}')
            self.train_qwk_history.append(train_qwk)

            running_loss = 0.
            all_y = []
            all_y_pred = []
            for x, y in self.valid_dataset_loader:
                x, y = Variable(x), Variable(y)
                if self.host_device == 'gpu':
                    x, y = x.cuda(), y.cuda()

                outputs = self.model(x)
                if self.host_device == 'gpu':
                    val_loss = self.loss_func(outputs.cuda(), y)
                else:
                    val_loss = self.loss_func(outputs, y)

                running_loss += val_loss.data[0]
                _, y_pred = torch.max(outputs.data, 1)
                all_y.append(y)
                all_y_pred.append(y_pred)

            y = torch.cat(all_y)
            y_pred = torch.cat(all_y_pred)
            val_qwk = quadratic_weighted_kappa(y_pred, y.data)

            logging.info(f'[Epoch {i_epoch+1}/{num_epochs}] '
                  f'VAL     QWK: {val_qwk:.3f}; loss: {running_loss / y.shape[0]:.3f}')

            self.val_qwk_history.append(val_qwk)
            self.val_loss_history.append(running_loss)
            training_time = time.time() - training_start_time
            logging.info(f"Epoch {i_epoch+1} - Training Time - {training_time} seconds")

            
            # scheduler step for plateau scheduler
            val_loss_scheduler = running_loss
            if (isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                self.scheduler.step(val_loss_scheduler)
            
            if val_qwk > self.best_qwk:
                logging.info(f'New best validation QWK score: {val_qwk}')
                self.best_qwk = val_qwk
                self.best_model = deepcopy(self.model)
                self.wait = 0
                logging.info('Storing best model...')
                torch.save(self.best_model, self.model_path)
                logging.info('Done storing')
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    logging.info('Stopped after epoch %d' % (i_epoch))
                    break

        training_time = time.time() - training_start_time
        logging.info(f"Full Training Time - {training_time} seconds")

    def predict_on_test(self):
        testing_start_time = time.time()
        all_y_pred = []
        all_image_names = []
        logging.info(f'num_test_images_batch={len(self.test_dataset_loader)}')
        image_index = 0
        bar = progressbar.ProgressBar(max_value=len(self.test_dataset_loader))
        bar.start(init=True)
        for x, image_names in self.test_dataset_loader:
            x = Variable(x)
            if self.host_device == 'gpu':
                x = x.cuda()
            outputs = self.model(x)
            _, y_pred = torch.max(outputs.data, 1)
            all_y_pred.append(y_pred)
            all_image_names.append(image_names)
            image_index += 1
            bar.update(image_index)
        bar.finish()

        all_image_names = list(chain.from_iterable(all_image_names))
        testing_time = time.time() - testing_start_time
        logging.info(f"Full Testing Time - {testing_time} seconds for "
                     f"{len(self.test_dataset_loader) * self.test_dataset_loader.batch_size} Images")
        if self.host_device == 'gpu':
            return torch.cat(all_y_pred).cpu().numpy(), all_image_names
        else:
            return torch.cat(all_y_pred).numpy(), all_image_names
