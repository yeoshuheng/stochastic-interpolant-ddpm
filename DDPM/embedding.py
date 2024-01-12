import torch.nn as nn
import math, torch

# Time embedding

# This is used by the model to consider the timestep of the model, this is important
# since parameters are shared across time.
# This is solved using positional embeddings by assigning a different vector to
# each index in time using cosine and sin index.
# This is added as a additional input apart from the noisy image in the model.

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -1 * embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings