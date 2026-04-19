import torch
import torch.nn as nn
from torchvision import models


class FramesModel(nn.Module):
    """
    Frame-level classification model using a ResNet-50 backbone.
    """

    def __init__(self, num_classes: int = 8, pretrained: bool = True, dropout: float = 0.4):
        """
        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): If True, load pretrained ImageNet weights.
            dropout (float): Dropout probability before the final layer.
        """
        super().__init__()

        # Load pretrained ResNet-50
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Get feature dimension
        in_features = self.backbone.fc.in_features

        # Replace final layer with a simple classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes).
        """
        return self.backbone(x)
