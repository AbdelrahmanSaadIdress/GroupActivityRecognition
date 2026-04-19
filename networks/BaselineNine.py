import torch
import torch.nn as nn
from torchvision import models

class HierarchicalModel(nn.Module):
    def __init__(self, person_num_classes=9, group_num_classes =8, hidden_size=512, num_layers=1  ):
        super(HierarchicalModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            *list(models.resnet34(weights=models.ResNet34_Weights.DEFAULT).children())[:-1]
        )
        self.gru_1 = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, person_num_classes)
        )
        self.pool = nn.AdaptiveMaxPool2d((1, 256))

        self.gru_2 = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5
        )
        
        self.fc_2 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, group_num_classes)
        )

    def forward(self, x):
        
        b, seq, bb, c, h, w = x.shape  # seq => frames
        x = x.permute(0,2,1,3,4,5).contiguous()
        x = x.view(b*bb*seq, c, h, w)  # (b * bb * seq, c, h, w)
        x1 = self.feature_extractor(x) # (batch * bbox * seq, 512, 1 , 1)

        x1 = x1.view(b*bb, seq, -1)       # (batch * bbox, seq, 512)
        x2, (h_1 , c_1) = self.gru_1(x1) # (batch * bbox, seq, hidden_size)
        y1 = self.fc_1(x2[:, -1, :])  # (batch, person_num_classes)

        x = torch.cat([x1, x2], dim=2)   
        x = x.contiguous()            

        x = x.view(b*seq, bb, -1) # (batch * seq, bbox, hidden_size)
        team_1 = x[:, :6, :]      # (batch * seq, 6, hidden_size)
        team_2 = x[:, 6:, :]      # (batch * seq, 6, hidden_size)
        team_1 = self.pool(team_1) # (batch * seq, 1, 256)
        team_2 = self.pool(team_2) # (batch * seq, 1, 256)

        x = torch.cat([team_1, team_2], dim=1)  # (batch * seq, 2, 256)
        x = x.view(b, seq, -1) # (batch, seq, 512)

        x, (h_2 , c_2) = self.gru_2(x) # (batch, seq, hidden_size)

        x = x[:, -1, :]     # (batch, hidden_size)
        y2 = self.fc_2(x)   # (batch, group_num_classes)
        return {'person_output': y1, 'group_output': y2}