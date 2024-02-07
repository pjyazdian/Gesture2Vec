import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math


class DAE_Network(nn.Module):
    def __init__(self, motion_dim, latent_dim):
        super(DAE_Network, self).__init__()
        print("init");
        self.encoder = nn.Sequential(
            nn.Linear(motion_dim, latent_dim),
            nn.Tanh(),
            # nn.Linear(motion_dim-10, motion_dim-30),
            # nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            # nn.Linear(motion_dim-30, motion_dim-10),
            # nn.Tanh(),
            nn.Linear(latent_dim, motion_dim),

        )

    def forward(self, x):
        # print("_________________")
        # print(self.encoder)
        # print(x.shape)
        # print("_________________")
        x = torch.squeeze(x)
        # print(x.shape)
        x = self.encoder(x)
        # print("Encoded", x.shape)
        x = self.decoder(x)
        x = torch.unsqueeze(x, 2)
        # print("Decoder", x.shape)
        return x