import torch
from torchvision import models

resnet_models = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152
}


class SiameseResnet(torch.nn.Module):
    def __init__(self, fc_layer, architecture='resnet18', pretrained=True):
        super(SiameseResnet, self).__init__()
        self.model = resnet_models[architecture](pretrained=pretrained)
        if fc_layer:
            self.model.fc = fc_layer(self.model.fc.in_features)

    def forward(self, *input):
        img1, img2 = input
        model = torch.nn.DataParallel(self.model)
        return model.forward(img1), model.forward(img2)

    def unfreeze_classifier(self):
        self.freeze_all()
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True