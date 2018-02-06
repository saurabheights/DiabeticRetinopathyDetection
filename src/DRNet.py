from torchvision.models import resnet18, resnet34, resnet50
import torch.nn as nn
from itertools import chain

class DRNet(nn.Module):
    
    '''
    Class to create a ResNet Model for transfer learning.
    
    Arguments:
    num_classes: number of output target classes, default is 5
    pretrained: boolean to determine whether to retrieve pretrained weights 'image-net', defaults to FALSE
    net_size: choose which ResNet architecture, can be 18, 34, 50, otherwise defaults to 50
    freeze_features: boolean to determine whether to freeze some layers or not, defaults to FALSE
    freeze_until_layer: integer to determine number of layers to freeze starting from the beginning of the network. 
    
    freeze_until_layer can be {1,2,3,4,5}, other values are not allowed.
    
    Each number corresponds to a layer in ResNet architecture, see Table in the paper: https://arxiv.org/pdf/1512.03385.pdf
    
    rates: an array of learning rates for each layer, has to be 6 elements (5 ResNet layers + fc layer)
    
    '''
    def __init__(self, num_classes=5, pretrained=False, net_size=50,
                 freeze_features=False, freeze_until_layer=5, rates=[], default_lr=1e-9):
        super(DRNet, self).__init__()
        
        self.layer_learning_rates = rates
        self.default_lr = default_lr
        
        if(net_size == 18):
            self.resnet = resnet18(pretrained=pretrained)
        elif (net_size == 34):
            self.resnet = resnet34(pretrained=pretrained)
        elif (net_size == 50):
            self.resnet = resnet50(pretrained=pretrained)
        else:
            print("Error in DRNet: Invalid model size for ResNet. Initializeing a ResNet 50 instead.")
            net_size = 50
            self.resnet = resnet50(pretrained=pretrained)
        
        layers = [self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool, #1
                 self.resnet.layer1, #2
                 self.resnet.layer2, #3
                 self.resnet.layer3, #4
                 self.resnet.layer4, #5
                 self.resnet.avgpool,
                 self.resnet.fc]
        
        if (freeze_features & pretrained):
            if (freeze_until_layer < 1) | (freeze_until_layer > 5):
                print("Error in DRNet: Freezing layers not possible. Cannot freeze parameters until the given layer.")
            else:
                for layer in layers[0:freeze_until_layer+3]:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
                
        self.set_name("DRNet(ResNet "+str(net_size)+")")
        
    def get_lr_layer(self, layer_index):
        if (len(self.layer_learning_rates) == 0):
            return self.default_lr
        
        if (layer_index < 1) | (layer_index > 6):
            return self.default_lr
        
        if (len(self.layer_learning_rates) < layer_index):
            return self.default_lr
            
        return self.layer_learning_rates[layer_index-1]
    
    def get_params_layer(self, layer_index):
        if (layer_index < 1) | (layer_index > 6):
            return None
        if (layer_index == 1):
            return chain(self.resnet.conv1.parameters(), self.resnet.bn1.parameters())
        if (layer_index == 2):
            return self.resnet.layer1.parameters()
        if (layer_index == 3):
            return self.resnet.layer2.parameters()
        if (layer_index == 4):
            return self.resnet.layer3.parameters()
        if (layer_index == 5):
            return self.resnet.layer4.parameters()
        if (layer_index == 6):
            return self.resnet.fc.parameters()
    
            
    def set_name(self, name):
        self.__name__ = name
        
    def forward(self, x):
        
        x = self.resnet(x)
        
        return x

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda