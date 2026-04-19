import torch
import torch.nn as nn
from torchvision import models

class OneCropNoSeqModel(nn.Module):
    def __init__(self, num_classes=9):
        super(OneCropNoSeqModel, self).__init__()
        # Load pretrained ResNet50
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        # Replace the final fully connected layer with a richer classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),   # shrink dimensions
            nn.BatchNorm1d(512),           # helps with stability
            nn.ReLU(),                     # nonlinearity
            nn.Dropout(0.5),               # regularization
            nn.Linear(512, num_classes)    # final classification
        )
    
    def forward(self, X):
        return self.model(X)
    

class WholeCropsNoSeqModel(nn.Module):
    def __init__(self, model = OneCropNoSeqModel(), num_classes = 8):
        super(WholeCropsNoSeqModel, self).__init__()
        # for name, param in model.model.named_parameters():
        #     if name.startswith("layer4"):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        
        # print(model)
        self.feature_extraction = nn.Sequential(*list(model.model.children())[:-1])
        self.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),

        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),

        nn.Linear(128, num_classes)
        )
        
    def forward(self, X):
        b, p, c, w, h = X.shape
        X = X.view(b*p, c, w, h)        # b*p, 3, 224, 224
        X = self.feature_extraction(X)  # b*p, 2048, 1, 1
        X = X.view(b, p, -1)            # b, p, 2048
        X = X.max(dim=1).values          # b, 2048
        X = self.fc(X)                   # b, num_classes

        return X
    

# WholeCropsNoSeqModel()
# print("fffff")