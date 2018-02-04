from datetime import datetime

import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau

import logging
import pprint
from sparse_config import data_params, train_data_transforms, val_data_transforms, test_data_transforms, \
    training_params, train_control, model_params, optimizer_params, kaggle_params
from data_loading import get_train_valid_loader, get_test_loader
from output_writing import write_submission_csv
from trainer import ModelTrainer
from kaggle import submit_solution

if __name__ == '__main__':
    # Check mode and model for logging file name
    mode = 'train' if model_params['train'] else 'test'
    model_name = model_params['model'].__name__

    # Handler - Basically output logging statements to both - a file and console.
    handlers = [logging.FileHandler(datetime.now().strftime(f"../logs/%Y-%m-%d_%H-%M-%S-{model_name}-{mode}.log")),
                logging.StreamHandler()]
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        level=logging.INFO, handlers=handlers)

    logging.info('Started')

    # Log training, data, model and optimizer parameters.
    logging.info(f"Training params:\n{pprint.pformat(training_params)}")
    logging.info(f"Training Control params:\n{pprint.pformat(train_control)}")
    logging.info(f"Data params:\n{pprint.pformat(data_params)}")
    logging.info(f"Model params:\n{pprint.pformat(model_params)}")
    logging.info(f"Optimizer params:\n{pprint.pformat(optimizer_params)}")

    train_dataset_loader, valid_dataset_loader = get_train_valid_loader(data_params['train_path'],
                                                                        data_params['label_path'],
                                                                        random_seed=54321,
                                                                        batch_size=data_params['batch_size'],
                                                                        rebalance_strategy=data_params['rebalance_strategy'],
                                                                        train_transforms=train_data_transforms,
                                                                        valid_transforms=val_data_transforms,
                                                                        num_workers=data_params['num_loading_workers'],
                                                                        pin_memory=False)
    test_dataset_loader = get_test_loader(data_params['test_path'],
                                          batch_size=data_params['batch_size'],
                                          transforms=test_data_transforms,
                                          num_workers=data_params['num_loading_workers'],
                                          pin_memory=False)

    
    if model_params['train'] and model_params['train_from_scratch']:
        model = model_params['model'](**model_params['model_kwargs'])
    else:
        if model_params['pytorch_device'] == 'gpu':
            model = torch.load(model_params['model_path'])
        else:
            model = torch.load(model_params['model_path'], lambda storage, loc: storage)

    
    # Pass only the trainable parameters to the optimizer, otherwise pyTorch throws an error
    # relevant to Transfer learning with fixed features
        
    optimizer = train_control['optimizer'](filter(lambda p: p.requires_grad, model.parameters()),
                                           **optimizer_params)
    
    
    # Initiate Scheduler 
    
    if (train_control['lr_scheduler_type'] == 'step'):
        scheduler = StepLR(optimizer, **train_control['step_scheduler_args'])
    elif (train_control['lr_scheduler_type'] == 'exp'):
        scheduler = ExponentialLR(optimizer, **train_control['exp_scheduler_args'])
    elif (train_control['lr_scheduler_type'] == 'plateau'):
        scheduler = ReduceLROnPlateau(optimizer, **train_control['plateau_scheduler_args'])
    else:
        scheduler = StepLR(optimizer, step_size=100, gamma = 1)
    
    if model_params['pytorch_device'] == 'gpu':
        with torch.cuda.device(model_params['cuda_device']):
            model_trainer = ModelTrainer(model, train_dataset_loader, valid_dataset_loader, test_dataset_loader,
                                         model_params['model_path'],
                                         optimizer = optimizer,
                                         optimizer_args=optimizer_params,
                                         scheduler = scheduler,
                                         host_device='gpu')
            if model_params['train']:
                model_trainer.train_model(**training_params)
            predictions, image_names = model_trainer.predict_on_test()

    else:
        model_trainer = ModelTrainer(model, train_dataset_loader, valid_dataset_loader, test_dataset_loader,
                                     model_params['model_path'],
                                     optimizer = optimizer,
                                     optimizer_args=optimizer_params,
                                     scheduler = scheduler,
                                     host_device='cpu')
        if model_params['train']:
            model_trainer.train_model(**training_params)
        predictions, image_names = model_trainer.predict_on_test()

    write_submission_csv(predictions, image_names, data_params['submission_file'])
    if kaggle_params['auto_submit'] :
        output = submit_solution(data_params['submission_file'])
        logging.info(f"Kaggle submission output = {output}")
    logging.info('Finished.')
