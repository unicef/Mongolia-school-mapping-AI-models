from torch import nn
from torchvision import models


class EfficientNet(nn.Module):
    
    def __init__(self, version, pretrained, num_classes):
        
        super(EfficientNet, self).__init__()
        
        self.model = self._get_model(version, pretrained)
        
        self.in_features = self.model.classifier[1].in_features
        
        self.model.classifier[1] = nn.Linear(in_features = self.in_features,
                                             out_features = num_classes,
                                             bias = True)
    
    
    def forward(self, x):
        
        x = self.model(x)
        
        return x
    
    
    def _get_model(self, version, pretrained):
        if version == "b0":
            return models.efficientnet_b0(pretrained=pretrained)
        elif version == "b1":
            return models.efficientnet_b1(pretrained=pretrained)
        elif version == "b2":
            return models.efficientnet_b2(pretrained=pretrained)
        elif version == "b3":
            return models.efficientnet_b3(pretrained=pretrained)
        elif version == "b4":
            return models.efficientnet_b4(pretrained=pretrained)
        elif version == "b5":
            return models.efficientnet_b5(pretrained=pretrained)
        elif version == "b6":
            return models.efficientnet_b6(pretrained=pretrained)
        elif version == "b7":
            return models.efficientnet_b7(pretrained=pretrained)