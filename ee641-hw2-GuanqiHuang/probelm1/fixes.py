import torch
import torch.nn as nn

class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, real_feats: torch.Tensor, fake_feats: torch.Tensor) -> torch.Tensor:
        return self.criterion(fake_feats.mean(dim=0), real_feats.mean(dim=0))

class DiscriminatorWithFeatures(nn.Module):
    def __init__(self, disc: nn.Module):
        super().__init__()
        self.disc = disc
        # split disc.net into backbone and head
        layers = list(self.disc.net.children())
        self.backbone = nn.Sequential(*layers[:-1])
        self.head = layers[-1]

    def features(self, x):
        h = self.backbone(x)
        return h

    def forward(self, x):
        h = self.features(x)
        logits = self.head(h)
        return logits, h
