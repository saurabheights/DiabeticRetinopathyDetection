from copy import deepcopy
from itertools import chain

import torch
from torch.autograd import Variable

from quadratic_weighted_kappa import quadratic_weighted_kappa


class ModelTrainer:
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, model, train_dataset_loader, valid_dataset_loader, test_dataset_loader,
                 host_device='cpu',
                 optimizer=torch.optim.Adam, optimizer_args={},
                 loss_func=torch.nn.CrossEntropyLoss(size_average=False),
                 patience=float('Inf')):
        self.model = model
        self.train_dataset_loader = train_dataset_loader
        self.valid_dataset_loader = valid_dataset_loader
        self.test_dataset_loader = test_dataset_loader

        self.host_device = host_device
        self.optimizer_args = optimizer_args
        self.optimizer = optimizer
        self.loss_func = loss_func

        self._reset_histories()
        self.patience = patience
        self.wait = 0
        self.best_qwk = 0
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

        optimizer = self.optimizer(self.model.parameters(), **self.optimizer_args)
        self._reset_histories()
        if self.host_device == 'gpu':
            self.model.cuda()
        iter_per_epoch = len(self.train_dataset_loader)
        print("Start training")
        for i_epoch in range(num_epochs):
            running_loss = 0.
            all_y = []
            all_y_pred = []
            for i_batch, (x, y) in enumerate(self.train_dataset_loader):
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
                    print(f'[Iteration {i_batch}/{iter_per_epoch}] '
                          f'TRAIN loss: {running_loss / sum(curr_y.shape[0] for curr_y in all_y):.3f}')
                self.train_loss_history.append(running_loss)
            y = torch.cat(all_y)
            y_pred = torch.cat(all_y_pred)
            train_qwk = quadratic_weighted_kappa(y_pred, y.data)

            print(f'[Epoch {i_epoch}/{num_epochs}] '
                  f'TRAIN   QWK: {train_qwk:.3f}; loss: {running_loss] / y.shape[0]:.3f}')
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

            print(f'[Epoch {i_epoch}/{num_epochs}] '
                  f'VAL     QWK: {val_qwk:.3f}; loss: {running_loss / y.shape[0]:.3f}')

            self.val_qwk_history.append(val_qwk)
            self.val_loss_history.append(running_loss)

            if val_qwk > self.best_qwk:
                self.best_qwk = val_qwk
                self.best_model = deepcopy(self.model)
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print('Stopped after epoch %d' % (i_epoch))
                    break

    def predict_on_test(self):
        all_y_pred = []
        all_image_names = []
        for x, image_names in self.test_dataset_loader:
            x = Variable(x)
            if self.host_device == 'gpu':
                x = x.cuda()
            outputs = self.model(x)
            _, y_pred = torch.max(outputs.data, 1)
            all_y_pred.append(y_pred)
            all_image_names.append(image_names)

        return torch.cat(all_y_pred).numpy(), list(chain.from_iterable(all_image_names))
