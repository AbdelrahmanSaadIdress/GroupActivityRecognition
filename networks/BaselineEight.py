# from .BaselineFive import SeqCropsModel
from .BaselineSeven import FeatureExtractor

import torch
import torch.nn as nn
from torchvision import models

class FullModelWithSorting(nn.Module):
    def __init__(self, prev_model:nn.Module=None, num_classes:int = 8 ):
        super(FullModelWithSorting, self).__init__()
                
        

        self.feature_extractor1 = prev_model.frame_feature_extractor
        self.lstm1 = prev_model.lstm

        for module in [self.feature_extractor1, self.lstm1]:
            for param in module.parameters():
                param.requires_grad = False

        # Max pooling across players (12 players → 1 vector per frame)
        self.max_pool = lambda x: x.max(dim=1)[0]

        # LSTM2: temporal modeling across frames (after pooling players)
        self.lstm2 = nn.LSTM(
            input_size=4096,
            hidden_size=2048,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, X):
        """
        X: (b, seq, p, c, w, h)
            b   = batch size
            seq = number of frames (e.g., 9)
            p   = number of players (12)
            c   = channels (3)
            w,h = frame size
        """
        b, seq, p, c, w, h = X.shape

        # --- Stage 1: CNN features for each player-frame ---
        X = X.permute(0, 2, 1, 3, 4, 5).contiguous()    # (b, p, seq, c, w, h)
        X = X.view(b * p * seq, c, w, h)                # (b*p*seq, c, w, h)
        X = self.feature_extractor1(X)                  # (b*p*seq, 2048, 1, 1)
        X = X.view(b * p, seq, -1)                      # (b*p, seq, 2048)

        # --- Stage 2: LSTM1 per player (temporal modeling across frames) ---
        out, _ = self.lstm1(X)                           # (b*p, seq, 1024)

        # --- Stage 3: Reshape to group players per frame ---
        out = out.view(b, p, seq, -1)                   # (b, p, seq, 1024)
        out = out.permute(0, 2, 1, 3).contiguous()      # (b, seq, p, 1024)
        out = out.view(b * seq, p, -1)                  # (b*seq, p, 1024)

        team1 = out[:,:6,:] ; team2 = out[:,6:,:]  

        team1 = self.max_pool(team1) ; team2 = self.max_pool(team2)        # b*9 , 1 , 1024 , b*9 , 1 , 1024
        out = torch.cat([team1, team2], axis = 1)    # b*9 , 2 , 1024
        out = out.view(b,seq, -1)               # b , 9 , 2048

        out, _ = self.lstm2(out)                # b , 9 , 1024
        out = out[:,-1,:]                       # b , 1024
        
        out = self.classifier(out)              # b , 8
        return out