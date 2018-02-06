"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak

Adjusted for ResNet architecture from
https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/cnn_layer_visualization.py

"""
import os

import cv2
import numpy as np
import torch
from torch.optim import SGD

from DRNet import DRNet
from visualization.misc_functions import preprocess_image, recreate_image

from torchvision.models.resnet import Bottleneck


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """

    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Create the folder to export images if not exists
        if not os.path.exists('../../generated'):
            os.makedirs('../../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        for module_name, module in self.model._modules.items():
            if module_name.startswith('layer'):
                for layer_module_name, layer_module in module._modules.items():
                    if module_name == self.selected_layer[0] and layer_module_name == self.selected_layer[1] \
                            and self.selected_layer[2] == 'conv1':
                        layer_module.conv1.register_forward_hook(hook_function)
                        break
                    if module_name == self.selected_layer[0] and layer_module_name == self.selected_layer[1] \
                            and self.selected_layer[2] == 'conv2':
                        layer_module.conv2.register_forward_hook(hook_function)
                        break
                    if module_name == self.selected_layer[0] and layer_module_name == self.selected_layer[1] \
                            and self.selected_layer[2] == 'conv3':
                        layer_module.conv3.register_forward_hook(hook_function)
                        break
            elif module_name != 'fc':
                if self.selected_layer == 'conv1' == module_name:
                    module.register_forward_hook(hook_function)
                    break

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image)
        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas later layers need less
        optimizer = SGD([self.processed_image], lr=5, weight_decay=1e-6)
        for i in range(1, 51):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image

            found_x = False
            self.model(x)
            for module_name, module in self.model._modules.items():
                if found_x:
                    break
                if module_name.startswith('layer'):
                    for layer_module_name, layer_module in module._modules.items():
                        residual = x
                        out = layer_module.conv1(x)
                        if module_name == self.selected_layer[0] and layer_module_name == self.selected_layer[1] \
                                and self.selected_layer[2] == 'conv1':
                            x = out
                            found_x = True
                            break
                        out = layer_module.bn1(out)
                        out = layer_module.relu(out)

                        out = layer_module.conv2(out)
                        if module_name == self.selected_layer[0] and layer_module_name == self.selected_layer[1] \
                                and self.selected_layer[2] == 'conv2':
                            x = out
                            found_x = True
                            break
                        out = layer_module.bn2(out)

                        if type(layer_module) == Bottleneck:
                            out = layer_module.relu(out)

                            out = layer_module.conv3(out)
                            if module_name == self.selected_layer[0] and layer_module_name == self.selected_layer[1] \
                                    and self.selected_layer[2] == 'conv3':
                                x = out
                                found_x = True
                                break
                            out = layer_module.bn3(out)

                        if layer_module.downsample is not None:
                            residual = layer_module.downsample(x)

                        out += residual
                        out = layer_module.relu(out)
                        x = out
                elif module_name != 'fc':
                    x = module(x)
                    if self.selected_layer == 'conv1' == module_name:
                        break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()[0]))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Save image
            if i % 50 == 0:
                cv2.imwrite('../../generated/resnet18_ft/layer_vis_l' + str(self.selected_layer) +
                            '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg',
                            self.created_image)

    def visualise_layer_without_hooks(self):
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image)
        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas later layers need less
        optimizer = SGD([self.processed_image], lr=5, weight_decay=1e-6)
        for i in range(1, 51):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image

            found_x = False
            for module_name, module in self.model._modules.items():
                if found_x:
                    break
                if module_name.startswith('layer'):
                    for layer_module_name, layer_module in module._modules.items():
                        residual = x
                        out = layer_module.conv1(x)
                        if module_name == self.selected_layer[0] and layer_module_name == self.selected_layer[1] \
                                and self.selected_layer[2] == 'conv1':
                            x = out
                            found_x = True
                            break
                        out = layer_module.bn1(out)
                        out = layer_module.relu(out)

                        out = layer_module.conv2(out)
                        if module_name == self.selected_layer[0] and layer_module_name == self.selected_layer[1] \
                                and self.selected_layer[2] == 'conv2':
                            x = out
                            found_x = True
                            break
                        out = layer_module.bn2(out)

                        if type(layer_module) == Bottleneck:
                            out = layer_module.relu(out)

                            out = layer_module.conv3(out)
                            if module_name == self.selected_layer[0] and layer_module_name == self.selected_layer[1] \
                                    and self.selected_layer[2] == 'conv3':
                                x = out
                                found_x = True
                                break
                            out = layer_module.bn3(out)

                        if layer_module.downsample is not None:
                            residual = layer_module.downsample(x)

                        out += residual
                        out = layer_module.relu(out)
                        x = out
                elif module_name != 'fc':
                    x = module(x)
                    if self.selected_layer == 'conv1' == module_name:
                        break

            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()[0]))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Save image
            if i % 50 == 0:
                cv2.imwrite('../../generated/layer_vis_l' + str(self.selected_layer) +
                            '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg',
                            self.created_image)


if __name__ == '__main__':
    # Set to 'conv1', ('layer1', '0', 'conv1'), ('layer1', '0', 'conv2'),...
    # ResNet18 and 34 have up to conv2, ResNet50 has conv3
    for cnn_layer in [
        'conv1',
        ('layer1', '0', 'conv1'), ('layer2', '1', 'conv2'), ('layer3', '0', 'conv1'),
                      # ('layer4', '2', 'conv3')
                          ('layer3', '1', 'conv2')
    ]:
        # Set to the number of the filter in the selectec conv layer
        for filter_pos in [0, 1, 2, 30]:
            pretrained_model = torch.load('../../models/DRNet_TL_18_Finetune_adam_plateau.model', lambda storage, loc: storage)
            if type(pretrained_model) == DRNet:
                pretrained_model = pretrained_model.resnet

            layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

            # Either way use visualize_layer_with_hooks or visualise_layer_without_hooks, both work fine.
            # Layer visualization with pytorch hooks
            layer_vis.visualise_layer_with_hooks()

            # Layer visualization without pytorch hooks
            # layer_vis.visualise_layer_without_hooks()
            print(f'Visualization done completed for {cnn_layer}, {filter_pos}')
