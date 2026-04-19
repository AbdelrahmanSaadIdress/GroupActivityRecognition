import torch
import torch.nn as nn
from torchvision import models
from .BaselineOne import FramesModel

class SeqFramesModel_V1(nn.Module):
    def __init__(self, model: nn.Module = FramesModel(), num_classes=8, lstm_hidden=1024, lstm_layers=1, dropout=0.5):
        super(SeqFramesModel_V1, self).__init__()

        # Freeze the feature extractor
        # for param in model.parameters():
        #     param.requires_grad = False

        # Frame-level feature extractor (remove final fc layer)
        self.frame_feature_extractor = nn.Sequential(*list(model.backbone.children())[:-1])  # b*seq, 2048,1,1

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        # Fully connected block for classification
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, X):
        """
        X: tensor of shape (batch, seq_len, channels, height, width)
        """
        b, seq, c, h, w = X.shape

        # Flatten batch and sequence for frame-level feature extraction
        X = X.view(b * seq, c, h, w)
        X = self.frame_feature_extractor(X)       # (b*seq, 2048, 1, 1)

        X = X.view(b, seq, -1)                    # (b, seq, 2048)


        # Pass through LSTM
        out, _ = self.lstm(X)                     # (b, seq, lstm_hidden)

        out = out[:, -1, :]                       # take last time step (b, lstm_hidden)


        # Classifier
        out = self.classifier(out)                # (b, num_classes)

        return out

class SeqFramesModel_V2(nn.Module):
    def __init__(self, model: nn.Module = FramesModel(), num_classes=8, lstm_hidden=1024, lstm_layers=1, dropout=0.5):
        super(SeqFramesModel_V2, self).__init__()

        # Freeze the feature extractor
        # for param in model.parameters():
        #     param.requires_grad = False

        # Frame-level feature extractor
        self.frame_feature_extractor = nn.Sequential(*list(model.backbone.children())[:-1])  # b*seq, 2048,1,1

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        # Fully connected block after concatenating skip connection
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden + 2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, X):
        """
        X: (batch, seq_len, channels, height, width)
        """
        b, seq, c, h, w = X.shape

        # Extract frame-level features
        X_frame = X.view(b * seq, c, h, w)
        X_frame = self.frame_feature_extractor(X_frame)      # (b*seq, 2048, 1, 1)
        X_frame = X_frame.view(b, seq, -1)                   # (b, seq, 2048)

        # Pass sequence through LSTM
        lstm_out, _ = self.lstm(X_frame)                    # (b, seq, lstm_hidden)
        lstm_out_last = lstm_out[:, -1, :]                  # (b, lstm_hidden)

        # Skip connection: concatenate last LSTM output with mean-pooled frame features
        skip_feat = X_frame.mean(dim=1)                     # (b, 2048)
        out = torch.cat([lstm_out_last, skip_feat], dim=1) # (b, lstm_hidden + 2048)

        # Classifier
        out = self.classifier(out)                          # (b, num_classes)
        return out