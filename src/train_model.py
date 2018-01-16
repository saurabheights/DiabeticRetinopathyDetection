from pprint import pprint

import torch

from configuration import data_params, train_data_transforms, val_data_transforms, test_data_transforms, training_params, model_params, optimizer_params
from data_loading import get_train_valid_loader, get_test_loader
from output_writing import write_submission_csv
from trainer import ModelTrainer

if __name__ == '__main__':
    print('Training params:')
    pprint(training_params)

    print("Data params:")
    pprint(data_params)

    print('Model params:')
    pprint(model_params)

    print('Optimizer params:')
    pprint(optimizer_params)

    train_dataset_loader, valid_dataset_loader = get_train_valid_loader(data_params['train_path'],
                                                                        data_params['label_path'],
                                                                        random_seed=54321,
                                                                        batch_size=data_params['batch_size'],
                                                                        rebalance_strategy=data_params['rebalance_strategy'],
                                                                        train_transforms=train_data_transforms,
                                                                        valid_transforms=val_data_transforms,
                                                                        num_workers=data_params['num_loading_workers'])
    test_dataset_loader = get_test_loader(data_params['test_path'],
                                          batch_size=data_params['batch_size'],
                                          transforms=test_data_transforms)

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
