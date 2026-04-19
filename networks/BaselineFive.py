from .BaselineThree import OneCropNoSeqModel
import torch.nn as nn
from torchvision import models

class SeqCropsModel(nn.Module):
    def __init__(self, prev_model : nn.Module = OneCropNoSeqModel(),  num_classes = 9):
        super(SeqCropsModel, self).__init__()

        # for param in prev_model.parameters():
        #     param.requires_grad = False
        
        self.frame_feature_extractor1 = nn.Sequential(*list(prev_model.model.children())[:-1])  
        
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=1024,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
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
        X = X.view( b*p, seq, -1 )              # b*12, 9, 2048
        out, _ = self.lstm(X)                   # b*12, 9, 1024
        out = out[:, -1, :]                     # b*12, 1024
        out = self.classifier(out)              # b*12, 9
        return out
    


class ClipsModel(nn.Module):
    """
    Improved ClipsModel:
        - Reuses SAME LSTM from SeqCropsModel (no new LSTM)
        - Adds small per-player MLP before pooling
        - Uses mean + max pooling instead of only max
        - Better classifier with LayerNorm
    """

    def __init__(self, prev_model, num_classes=8):
        super().__init__()

        # REUSE components from SeqCropsModel
        self.feature_extractor1 = prev_model.frame_feature_extractor1   # SAME CNN
        self.lstm = prev_model.lstm                                    # SAME LSTM

        feature_dim = 2048
        lstm_dim = 2048   

        # ---- small MLP to refine player features BEFORE pooling ----
        self.player_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # ---- improved classifier ----
        self.classifier = nn.Sequential(
            nn.Linear(lstm_dim, 512, bias=False),
            nn.LayerNorm(512),              # better than BatchNorm here
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, X):
        b, seq, p, c, w, h = X.shape

        # CNN
        X = X.view(b * seq * p, c, w, h)
        X = self.feature_extractor1(X)        # (b*seq*p, 2048, 1, 1)
        X = X.view(b, seq, p, -1)             # (b, seq, p, 2048)

        # refine per-player
        X = self.player_mlp(X)                # (b, seq, p, 2048)

        # pooling over players (p)
        mean_pool = X.mean(dim=2)             # (b, seq, 2048)
        max_pool  = X.max(dim=2).values       # (b, seq, 2048)
        frame_embed = mean_pool + max_pool    # (b, seq, 2048)

        # SAME LSTM (operates over (b, seq, 2048))
        out, _ = self.lstm(frame_embed)       # (b, seq, 1024)
        out = out[:, -1, :]                   # (b, 1024)

        # classifier
        out = self.classifier(out)            # (b, num_classes)
        return out
