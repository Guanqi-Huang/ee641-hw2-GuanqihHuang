from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
INPUT_T = 16
INPUT_C = 9
INPUT_D = INPUT_T * INPUT_C

class Encoder(nn.Module):
    def __init__(self, z_high_dim: int = 8, z_low_dim: int = 16, hidden: int = 256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.backbone = nn.Sequential(
            nn.Linear(INPUT_D, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # high-level posterior q(z_h|x)
        self.mu_h   = nn.Linear(hidden, z_high_dim)
        self.logv_h = nn.Linear(hidden, z_high_dim)
        # low-level posterior q(z_l|x, z_h)
        self.low_pre = nn.Sequential(
            nn.Linear(hidden + z_high_dim, hidden), nn.ReLU()
        )
        self.mu_l   = nn.Linear(hidden, z_low_dim)
        self.logv_l = nn.Linear(hidden, z_low_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        h = self.backbone(self.flatten(x))
        mu_h, logv_h = self.mu_h(h), self.logv_h(h)

        eps_h = torch.randn_like(mu_h)
        z_h   = mu_h + torch.exp(0.5 * logv_h) * eps_h

        h_low_in = torch.cat([h, z_h], dim=-1)
        h_low = self.low_pre(h_low_in)
        mu_l, logv_l = self.mu_l(h_low), self.logv_l(h_low)

        eps_l = torch.randn_like(mu_l)
        z_l   = mu_l + torch.exp(0.5 * logv_l) * eps_l
        return z_h, mu_h, logv_h, z_l, mu_l, logv_l

class Decoder(nn.Module):
    def __init__(self, z_high_dim: int = 8, z_low_dim: int = 16, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_high_dim + z_low_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, INPUT_D)  
        )

    def forward(self, z_h: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
        logits = self.net(torch.cat([z_h, z_l], dim=-1))  
        return logits.view(-1, INPUT_T, INPUT_C)

class HierarchicalVAE(nn.Module):
    def __init__(self, z_high_dim=8, z_low_dim=16, hidden=256):
        super().__init__()
        self.enc = Encoder(z_high_dim, z_low_dim, hidden)
        self.dec = Decoder(z_high_dim, z_low_dim, hidden)

    def forward(self, x):
        z_h, mu_h, logv_h, z_l, mu_l, logv_l = self.enc(x)
        logits = self.dec(z_h, z_l)
        return logits, (mu_l, logv_l, mu_h, logv_h)

    @staticmethod
    def kl_standard_normal(mu, logvar):
        # KL[N(mu, diag(exp(logvar))) || N(0,I)]
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    def elbo(self, x, logits, mu_l, logv_l, mu_h, logv_h, beta=1.0):
        # Reconstruction: BCE with logits
        recon = F.binary_cross_entropy_with_logits(logits, x, reduction='none')
        recon = recon.sum(dim=(1,2))  
        # KLs
        kl_h = self.kl_standard_normal(mu_h, logv_h)  
        kl_l = self.kl_standard_normal(mu_l, logv_l) 
        loss = recon + beta * (kl_h + kl_l)
        return loss.mean(), recon.mean(), kl_l.mean(), kl_h.mean()
