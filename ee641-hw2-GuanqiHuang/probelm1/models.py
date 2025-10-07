import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

class Generator(nn.Module):
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
            View((-1, 128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),   
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh(),  # [-1, 1]
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),      
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),     
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 2, 1),    
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            View((-1, 128*4*4)),
            nn.Linear(128*4*4, 1),
        )

    def forward(self, x):
        return self.net(x)
