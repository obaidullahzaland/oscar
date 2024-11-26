from torchvision import models, transforms
import torch.nn as nn
import torch

def initialize_model(backbone, num_classes, pretrained=True):
    if backbone == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif backbone == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif backbone == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif backbone == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif backbone == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif backbone == 'vit_b_16':
        model = models.vit_b_16(pretrained=pretrained)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
    elif backbone == 'vit_b_32':
        model = models.vit_b_32(pretrained=pretrained)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Invalid backbone model name")
    return model


# Define the model
class ServerTune(nn.Module):
    def __init__(self, classes=345):
        super(ServerTune, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()
        self.final_proj = nn.Sequential(
            nn.Linear(512, classes)
        )
    
    def forward(self, x, get_fea=False, input_image=True):
        if input_image:
            with torch.no_grad():  # Freeze encoder during forward pass
                x = self.encoder(x)
        
        if get_fea:  # Return extracted features
            return x.view(x.shape[0], -1)
        
        # Pass features to the final projection
        out = self.final_proj(x.view(x.shape[0], -1))
        return out