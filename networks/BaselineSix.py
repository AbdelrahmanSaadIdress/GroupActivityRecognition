from .BaselineThree import OneCropNoSeqModel

import torch
import torch.nn as nn
from torchvision import models

class CropsModel(nn.Module):
    def __init__(self, prev_model : nn.Module = OneCropNoSeqModel(),  num_classes = 9):
        super(CropsModel, self).__init__()

        for name, param in prev_model.named_parameters():
            # if name.startswith(("layer3", "layer4")):
            param.requires_grad = True
            # else :
            #     param.requires_grad = False
        
        self.frame_feature_extractor1 = nn.Sequential(*list(prev_model.model.children())[:-1])  

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.5),
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, X):
        b, seq, p, c, w, h = X.shape
        X = X.permute(0, 2, 1, 3, 4, 5).contiguous()
        X = X.view(b*seq*p, c, w, h)
        X = self.frame_feature_extractor1(X)    # b*12*9, 2048, 1, 1
        X = X.view( b*p*seq, -1 )               # b*12*9, 2048
        out = self.classifier(X)                # b*12*9 , 9
        return out
    

class SeqFramesModel(nn.Module):
    def __init__(self, prev_model: nn.Module, num_classes=8):
        super(SeqFramesModel, self).__init__()

        # Freeze pretrained weights
        for param in prev_model.parameters():
            param.requires_grad = True


        # Feature extractor without final FC
        self.frame_feature_extractor1 = prev_model.frame_feature_extractor1

        # Normalize extracted features
        self.feature_norm = nn.LayerNorm(2048)

        # Sequence model
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.5)

        # Attention pooling over time
        self.attention = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(1024),                 
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, X):
        b, seq, p, c, w, h = X.shape
        X = X.view(b * seq * p, c, w, h)

        # Feature extraction
        X = self.frame_feature_extractor1(X)       # (b*seq*p, 2048, 1, 1)
        X = X.view(b * seq, p, -1)                 # (b*seq, p, 2048)
        X = X.max(dim=1)[0]                        # (b*seq, 2048)
        X = self.feature_norm(X)                  
        X = X.view(b, seq, -1)                     # (b, seq, 2048)

        # Sequence modeling
        out, _ = self.lstm(X)                      # (b, seq, 1024)

        # Attention pooling (learn weights for frames)
        attn_weights = torch.softmax(self.attention(out), dim=1)  # (b, seq, 1)
        out = torch.sum(attn_weights * out, dim=1)                # (b, 1024)

        out = self.dropout(out)

        # Classification
        out = self.classifier(out)
        return out