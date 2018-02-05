"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak

Adjusted for ResNet architecture from
https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/gradcam.py
"""
import cv2
import numpy as np
import torch
from torchvision.models.resnet import Bottleneck

from visualization.misc_functions import get_params, save_class_activation_on_image


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_name, module in self.model._modules.items():
            if module_name.startswith('layer'):
                for layer_module_name, layer_module in module._modules.items():
                    residual = x
                    out = layer_module.conv1(x)
                    if module_name == self.target_layer[0] and layer_module_name == self.target_layer[1] \
                            and self.target_layer[2] == 'conv1':
                        out.register_hook(self.save_gradient)
                        conv_output = out
                    out = layer_module.bn1(out)
                    out = layer_module.relu(out)

                    out = layer_module.conv2(out)
                    if module_name == self.target_layer[0] and layer_module_name == self.target_layer[1] \
                            and self.target_layer[2] == 'conv2':
                        out.register_hook(self.save_gradient)
                        conv_output = out
                    out = layer_module.bn2(out)

                    if type(layer_module) == Bottleneck:
                        out = layer_module.relu(out)

                        out = layer_module.conv3(out)
                        if module_name == self.target_layer[0] and layer_module_name == self.target_layer[1] \
                                and self.target_layer[2] == 'conv3':
                            out.register_hook(self.save_gradient)
                            conv_output = out
                        out = layer_module.bn3(out)

                    if layer_module.downsample is not None:
                        residual = layer_module.downsample(x)

                    out += residual
                    out = layer_module.relu(out)
                    x = out
            elif module_name != 'fc':
                x = module(x)
                if self.target_layer == 'conv1' == module_name:
                    x.register_hook(self.save_gradient)
                    conv_output = x
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.fc(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_index=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_index is None:
            target_index = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_index] = 1
        # Zero grads
        self.model.zero_grad()
        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam


if __name__ == '__main__':
    # one for each class, you might adjust the paths in misc_functions#get_params
    for target_example in range(5):
        (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
            get_params(target_example)
        # Set to 'conv1', ('layer1', '0', 'conv1'), ('layer1', '0', 'conv2'),...
        for target_layer in [
            'conv1', ('layer1', '0', 'conv1'), ('layer2', '1', 'conv2'), ('layer3', '0', 'conv1'),
                             # ('layer4', '1', 'conv2')
                          ('layer4', '2', 'conv3')
                             ]:
            # Grad cam
            grad_cam = GradCam(pretrained_model, target_layer=target_layer)
            # Generate cam mask
            cam = grad_cam.generate_cam(prep_img, target_class)
            # Save mask
            current_file_name_to_export = file_name_to_export + f'_{str(target_layer)}'
            save_class_activation_on_image(original_image, cam, current_file_name_to_export)
            print(f'Grad cam completed for {target_example}, {target_layer}')
