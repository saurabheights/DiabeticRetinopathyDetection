# Configuration file with custom transforms for train/test/valid

from math import tan
from random import uniform

from PIL import Image
from torchvision import transforms
from torchvision.models import AlexNet

data_params = {
    'train_path': '../data/small_train_300',
    'test_path': '../data/test',
    'label_path': '../data/trainLabels.csv',
    'batch_size': 5,
    'submission_file': '../data/submission.csv',
    # 'even', 'posneg', None.
    # 'even': Same number of samples for each class
    # 'posneg': Same number of samples for class 0 and all other classes
    'rebalance_strategy': 'even',
    'num_loading_workers': 4
}

training_params = {
    'num_epochs': 50,
    'log_nth': 5
}

model_params = {
    # if False, just load the model from the disk and evaluate
    'train': True,
    'model_path': '../models/alexnet.model',
    'model': AlexNet,
    'model_kwargs': {'num_classes': 5},
    # the device to put the variables on (cpu/gpu)
    'pytorch_device': 'cpu',
    # cuda device if gpu
    'cuda_device': 0,
}

optimizer_params = {
    'lr': 1e-4
}

# training data transforms (random rotation, random skew, scale and crop 224)
train_data_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.rotate(uniform(0, 360), resample=Image.BICUBIC)),  # random rotation 0 to 360
    transforms.Lambda(lambda x: skew_image(x, uniform(-0.2, 0.2), inc_width=True)),  # random skew +- 0.2
    transforms.RandomResizedCrop(300, scale=(0.9, 1.1), ratio=(1, 1)),  # scale +- 10%, resize to 300
    transforms.CenterCrop((224)),
    transforms.ToTensor()
])

# validation data transforms (random rotation, random skew, scale and crop 224)
val_data_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.rotate(uniform(0, 360), resample=Image.BICUBIC)),  # random rotation 0 to 360
    transforms.Lambda(lambda x: skew_image(x, uniform(-0.2, 0.2), inc_width=True)),  # random skew +- 0.2
    transforms.RandomResizedCrop(300, scale=(0.9, 1.1), ratio=(1, 1)),  # scale +- 10%, resize to 300
    transforms.CenterCrop((224)),
    transforms.ToTensor()
])

# test data transforms (random rotation)
test_data_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.rotate(uniform(0, 360), resample=Image.BICUBIC)),  # random rotation 0 to 360
    transforms.RandomResizedCrop(300, scale=(1, 1), ratio=(1, 1)),  # resize to 300
    transforms.CenterCrop((224)),
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
