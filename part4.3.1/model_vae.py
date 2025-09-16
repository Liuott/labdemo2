# part4_vae/model_vae.py
import torch, torch.nn as nn, torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1,32,3,2,1), nn.ReLU(),   # 64
            nn.Conv2d(32,64,3,2,1), nn.ReLU(),  # 32
            nn.Conv2d(64,128,3,2,1), nn.ReLU(), # 16
            nn.Conv2d(128,256,3,2,1), nn.ReLU() # 8
        )
        self.fc_mu = nn.Linear(256*8*8, z_dim)
        self.fc_lv = nn.Linear(256*8*8, z_dim)
        self.fc_z  = nn.Linear(z_dim, 256*8*8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(), # 16
            nn.ConvTranspose2d(128,64,4,2,1),  nn.ReLU(), # 32
            nn.ConvTranspose2d(64,32,4,2,1),   nn.ReLU(), # 64
            nn.ConvTranspose2d(32,1,4,2,1),    nn.Sigmoid() # 128
        )

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_lv(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar); eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.fc_z(z).view(-1,256,8,8)
        return self.dec(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        xhat = self.decode(z)
        return xhat, mu, lv

def vae_loss(x, xhat, mu, logvar, eps=1e-6):
    # 防止 log(0) / log(1)
    xhat = xhat.clamp(eps, 1 - eps).float()
    x    = x.float()
    rec = F.binary_cross_entropy(xhat, x, reduction='sum')
    kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + kl, rec, kl
