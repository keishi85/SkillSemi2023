import torch.nn as nn
import torch.optim as optim
from torchvision import models



class Vgg16():
    def __init__(self):
        self.net = models.vgg16(pretrained=True)
        print(self.net)

        # Convert output layer into two
        self.net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

        # Define loss function
        self.loss_func = nn.CrossEntropyLoss()

        # Specify the model weights to update for each layer
        # Feature module
        params_to_update_1 = []
        # classifier module
        params_to_update_2 = []
        # last classifier module
        params_to_update_3 = []

        # Specify the name of layers when training
        update_param_names_1 = ['features']
        update_param_names_2 = ['classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias']
        update_param_names_3 = ['classifier.6.weight', 'classifier.6.bias']

        # Store each parameter in each list
        for name, param in self.net.named_parameters():
            # Update parameters
            param.requires_grad = True

            if update_param_names_1[0] in name:
                params_to_update_1.append(param)
            elif name in update_param_names_2:
                params_to_update_2.append(param)
            elif name in update_param_names_3:
                params_to_update_3.append(param)
            else:
                print(f'{name} does not apply to any')

        # Define optimization method
        self.optimizer = optim.SGD([
            {'params': params_to_update_1, 'lr': 1e-4},
            {'params': params_to_update_2, 'lr': 5e-4},
            {'params': params_to_update_3, 'lr': 1e-3},
        ], momentum=0.9)

    # Change optimization method
    def change_opt(self, method='Adam', beta_1=0.09, lr=0.01):
        if method == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(),
                                   lr=lr,
                                   betas=(beta_1, 0.999))



if __name__ == '__main__':
    vgg = Vgg16()