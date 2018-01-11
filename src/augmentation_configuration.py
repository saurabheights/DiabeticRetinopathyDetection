# Configuration file with custom transforms for train/test/valid

from torchvision import transforms
from torchvision.models import AlexNet
from random import uniform
import numpy as np
from PIL import Image
from math import tan

data_params = {
    'train_path': '../data/sample_300',
    'test_path': '../data/test',
    'label_path': '../data/trainLabels.csv',
    'batch_size': 1,
    'submission_file': '../data/submission.csv'
}

training_params = {
    'num_epochs': 20,
    'log_nth': 1
}

model_params = {
    'model': AlexNet,
    'model_kwargs': {'num_classes': 5},
    # the device to put the variables on (cpu/gpu)
    'pytorch_device': 'cpu',
    # cuda device if gpu
    'cuda_device': 0,
}

optimizer_params = {
    'lr': 1e-3
}

# training data transforms (random rotation, random skew, scale and crop 224)
train_data_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.rotate(uniform(0,360))),
    transforms.Lambda(lambda x: skew_image(x, uniform(-0.2, 0.2), inc_width=True)),
    transforms.Scale(224),
    transforms.RandomSizedCrop(224),
    transforms.ToTensor()
])

# validation data transforms (random rotation, random skew, scale and crop 224)
val_data_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.rotate(uniform(0,360))),
    transforms.Lambda(lambda x: skew_image(x, uniform(-0.2, 0.2), inc_width=True)),
    transforms.Scale(224),
    transforms.RandomSizedCrop(224),
    transforms.ToTensor()
])

# test data transforms (random rotation)
test_data_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.rotate(uniform(0,360))),
    transforms.Scale(224),
    transforms.RandomSizedCrop(224),
    transforms.ToTensor()
])


# Function to implement skew based on PIL transform

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
    
    