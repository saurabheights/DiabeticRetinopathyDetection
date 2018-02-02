# create a configuration.py like this
# transfer learning configuration file Example

from math import tan
from random import uniform

from PIL import Image
from torchvision import transforms
import torch.nn as nn
from DRNet import DRNet
from torch.optim import Adam, SGD


data_params = {
    'train_path': '../data/train_300',
    'test_path': '../data/test',
    'label_path': '../data/trainLabels.csv',
    'batch_size': 64,
    'submission_file': '../data/submission.csv',
    # 'even', 'posneg', None.
    # 'even': Same number of samples for each class
    # 'posneg': Same number of samples for class 0 and all other classes
    'rebalance_strategy': 'even',
    'num_loading_workers': 8
}

kaggle_params = {
    # Change auto submit to True, to submit from code.
    'auto_submit':False,
    # Change to your Kaggle username and password.
    'kaggle_username':'abc',
    'kaggle_password':'xyz',
    # Keep message enclosed in single qoutes, which are further enclosed in double qoutes.
    'kaggle_submission_message':"'Luke, I am you father'"
}

training_params = {
    'num_epochs': 50,
    'log_nth': 10,
}

train_control = {
    'optimizer' : Adam,             # Adam, SGD (we can add more)
    'lr_scheduler_type': 'plateau',    # 'exp', 'step', 'plateau', 'none'
    
    'step_scheduler_args' : {
        'gamma' : 0.1,       # factor to decay learing rate (new_lr = gamma * lr)
        'step_size': 3     # number of epochs to take a step of decay
    },

    'exp_scheduler_args' : {
        'gamma' : 0.1       # factor to decay learing rate (new_lr = gamma * lr)    
    },

    'plateau_scheduler_args' : {
        'factor' : 0.1,      # factor to decay learing rate (new_lr = factor * lr)   
        'patience' : 5,     # number of epochs to wait as monitored value does not change before decreasing LR 
        'verbose' : True,    # print a message when LR is changed
        'threshold' : 1e-3,  # when to consider the monitored varaible not changing (focus on significant changes)
        'min_lr' : 1e-9,     # lower bound on learning rate, not decreased further
        'cooldown' : 0       # number of epochs to wait before resuming operation after LR was reduced
    }

}

optimizer_params = {
    'lr': 1e-3
}



model_params = {
    # if False, just load the model from the disk and evaluate
    'train': True,
    # if False, previously (partially) trained model is further trained.
    'train_from_scratch': True,
    'model_path': '../models/DRNet_TL_varLR_adam_plateau.model',
    'model': DRNet,
    'per_layer_rates' : True,   # if True, the array rates in model kwargs will be used
    'model_kwargs' : {
        'num_classes' : 5,
        'pretrained' : True,        # load pre-trained weights on image-net
        'net_size' : 18,            # 18, 34, or 50
        'freeze_features' : False,   # fixed feature extractor OR fine-tune
        'freeze_until_layer' : 2,    # 1, 2, 3, 4, or 5 (check ResNet Paper)
        'rates' : [1e-5, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3]                
        # array for learning rates of DRNet, layers 1 to 5 ResNet, 6 fc classifier
    },
    # the device to put the variables on (cpu/gpu)
    'pytorch_device': 'gpu',
    # cuda device if gpu
    'cuda_device': 0,
}




# normalization recommended by PyTorch documentation
# only in case of transfer learning

transfer_learn = model_params['model_kwargs']['pretrained']

normalize_transfer_learning = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if transfer_learn else transforms.Normalize(
    mean=[0, 0, 0], std=[1, 1, 1])


# training data transforms (random rotation, random skew, scale and crop 224)
train_data_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.rotate(uniform(0,360), resample=Image.BICUBIC)),  # random rotation 0 to 360
    transforms.Lambda(lambda x: skew_image(x, uniform(-0.2, 0.2), inc_width=True)), # random skew +- 0.2
    transforms.RandomResizedCrop(300, scale=(0.9, 1.1), ratio=(1,1)),               # scale +- 10%, resize to 300
    transforms.CenterCrop((224)),
    transforms.ToTensor(),
    normalize_transfer_learning
])

# validation data transforms (random rotation, random skew, scale and crop 224)
val_data_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.rotate(uniform(0,360), resample=Image.BICUBIC)),  # random rotation 0 to 360
    transforms.Lambda(lambda x: skew_image(x, uniform(-0.2, 0.2), inc_width=True)), # random skew +- 0.2
    transforms.RandomResizedCrop(300, scale=(0.9, 1.1), ratio=(1,1)),               # scale +- 10%, resize to 300
    transforms.CenterCrop((224)),
    transforms.ToTensor(),
    normalize_transfer_learning
])

# test data transforms (random rotation)
test_data_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.rotate(uniform(0,360), resample=Image.BICUBIC)),  # random rotation 0 to 360
    transforms.RandomResizedCrop(300, scale=(1,1), ratio=(1,1)),                    # resize to 300
    transforms.CenterCrop((224)),
    transforms.ToTensor(),
    normalize_transfer_learning
])


# Function to implement skew based on PIL transform
# adopted from: https://www.programcreek.com/python/example/69877/PIL.Image.AFFINE

def skew_image(img, angle, inc_width=False):
    """
    Skew image using some math
    :param img: PIL image object
    :param angle: Angle in radians (function doesn't do well outside the range -1 -> 1, but still works)
    :return: PIL image object
    """
    width, height = img.size
    # Get the width that is to be added to the image based on the angle of skew
    xshift = tan(abs(angle)) * height
    new_width = width + int(xshift)

    if new_width < 0:
        return img

    # Apply transform
    img = img.transform(
        (new_width, height),
        Image.AFFINE,
        (1, angle, -xshift if angle > 0 else 0, 0, 1, 0),
        Image.BICUBIC
    )

    if (inc_width):
        return img
    else:
        return img.crop((0, 0, width, height))

