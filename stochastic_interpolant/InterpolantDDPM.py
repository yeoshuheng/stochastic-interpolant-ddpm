import torch.nn as nn
from DDPM.ddpm import DDPM

class InterpolantDDPM(nn.modules):
    def __init__(self):
        self.v_net = DDPM() # Velocity field model
        self.n_net = DDPM() # Noise model
        self.s_net = DDPM() # Score model
        self.b_net = DDPM() # Velocity model