from datetime import datetime

import torch
import logging
import pprint
from configuration import data_params, train_data_transforms, val_data_transforms, test_data_transforms, \
    training_params, model_params, optimizer_params
from data_loading import get_train_valid_loader, get_test_loader
from output_writing import write_submission_csv
from trainer import ModelTrainer

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

    if model_params['train']:
        model = model_params['model'](**model_params['model_kwargs'])
    else:
        model = torch.load(model_params['model_path'])

    if model_params['pytorch_device'] == 'gpu':
        with torch.cuda.device(model_params['cuda_device']):
            model_trainer = ModelTrainer(model, train_dataset_loader, valid_dataset_loader, test_dataset_loader,
                                         model_params['model_path'],
                                         optimizer_args=optimizer_params,
                                         host_device='gpu')
            if model_params['train']:
                model_trainer.train_model(**training_params)
            predictions, image_names = model_trainer.predict_on_test()

    else:
        model_trainer = ModelTrainer(model, train_dataset_loader, valid_dataset_loader, test_dataset_loader,
                                     model_params['model_path'],
                                     optimizer_args=optimizer_params,
                                     host_device='cpu')
        if model_params['train']:
            model_trainer.train_model(**training_params)
        predictions, image_names = model_trainer.predict_on_test()

    write_submission_csv(predictions, image_names, data_params['submission_file'])
    logging.info('Finished.')
