# from .BaselineFive import SeqCropsModel

import torch
import torch.nn as nn
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self, num_classes = 9):
        super(FeatureExtractor, self).__init__()

        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )
        self.frame_feature_extractor = nn.Sequential(*list(backbone.children())[:-1]) 

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
        X = self.frame_feature_extractor(X)     # b*12*9, 2048, 1, 1
        X = X.view( b*p, seq, -1 )              # b*12, 9, 2048
        out, _ = self.lstm(X)                   # b*12, 9, 1024
        out = out[:, -1, :]                     # b*12, 1024
        out = self.classifier(out)              # b*12, 9
        return out
    


class FullModelWithNoSorting(nn.Module):
    """
    Full group activity classifier.
    """
    def __init__(self, person_feature_extraction: nn.Module, hidden_size:int=1024, num_layers:int=1, num_classes: int = 8):
        super().__init__()

        # ---------------------------------------
        # Pretrained per-person feature extractor
        # ---------------------------------------
        self.resnet50 = person_feature_extraction.frame_feature_extractor
        self.lstm_1 = person_feature_extraction.lstm

        # Freeze pretrained modules
        for module in [self.resnet50, self.lstm_1]:
            for param in module.parameters():
                param.requires_grad = False

        # -----------------------------
        # Projection for concatenated features (CNN + LSTM1)
        # -----------------------------
        self.proj = nn.Linear(2048 + 2048, 2048)  # CNN (2048) + LSTM1 (bidirectional=2048) → 2048

        # -----------------------------
        # Pooling over players
        # -----------------------------
        self.pool = nn.AdaptiveMaxPool1d(1)  # pool over player dimension

        # -----------------------------
        # Normalization
        # -----------------------------
        self.layer_norm = nn.LayerNorm(2048)

        # -----------------------------
        # LSTM_2: scene-level temporal modeling
        # -----------------------------
        self.lstm_2 = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # -----------------------------
        # Classifier head
        # -----------------------------
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, people, frames, channels, H, W)
        b, bb, seq, c, h, w = x.shape

        # -----------------------------
        # 1) CNN feature extraction per person-frame
        # -----------------------------
        x = x.view(b * bb * seq, c, h, w)
        cnn_feats = self.resnet50(x)                   # (b*bb*seq, 2048, 1, 1)
        cnn_feats = cnn_feats.view(b * bb, seq, -1)   # (b*bb, seq, 2048)
        cnn_feats = self.layer_norm(cnn_feats)

        # -----------------------------
        # 2) LSTM_1: temporal modeling per person
        # -----------------------------
        lstm1_out, _ = self.lstm_1(cnn_feats)         # (b*bb, seq, 2048)

        # -----------------------------
        # 3) Concatenate CNN + LSTM1 features
        # -----------------------------
        person_feats = torch.cat([cnn_feats, lstm1_out], dim=2)  # (b*bb, seq, 4096)
        person_feats = self.proj(person_feats)                   # (b*bb, seq, 2048)

        # -----------------------------
        # 4) Pool over players
        # -----------------------------
        person_feats = person_feats.view(b * seq, bb, -1)        # (b*seq, num_players, 2048)
        person_feats = person_feats.permute(0, 2, 1)             # (b*seq, 2048, num_players)
        pooled = self.pool(person_feats).squeeze(-1)             # (b*seq, 2048)

        # -----------------------------
        # 5) Prepare for LSTM_2
        # -----------------------------
        pooled = pooled.view(b, seq, -1)                         # (b, seq, 2048)
        pooled = self.layer_norm(pooled)

        # -----------------------------
        # 6) LSTM_2: scene-level temporal modeling
        # -----------------------------
        out, _ = self.lstm_2(pooled)                             # (b, seq, hidden_size)

        # -----------------------------
        # 7) Classification: last time step
        # -----------------------------
        out = self.fc(out[:, -1, :])                             # (b, num_classes)

        return out