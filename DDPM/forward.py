import torch
import torch.nn.functional as F
from config import CONFIG

# Forward Process

# Product of conditional distribution with a mean that depends on the previous timestep and a
# specific variance that depends on the scheduler.
# The noisy version of a image at any timestep, x[t] can also be directly calculated from x[0]
# as all process is gaussian. 
# Variance is re-parameterised as alpha for easier closed-form calculation.

def linear_beta_schedule(t, start=0.0001, end=0.02):
    return torch.linspace(start, end, t)

T = CONFIG["total_timesteps"]
beta = linear_beta_schedule(T)
alpha = 1 - beta
alpha_cumprod = torch.cumprod(alpha, axis =0)
alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alpha = torch.sqrt(1 / alpha)
sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
posterior_variance = beta * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)


def helper(vals, t, x_shape):
    """
    @param vals: Values.
    @param x_shape: Shape of the values to obtain batch dimensions.
    @param t : Index to extract.

    @return Value[i] while considering batch size.
    """
    batch_size = t.shape[0]
    # Gathers among axis specified by a single dimension.
    # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion(x_0, t, device="cpu"):
    """
    @param t: Timestep.

    @return Noisy version of image at t.
    """
    noise = torch.rand_like(x_0)
    sqrt_alpha_cumprod_t = helper(sqrt_alpha_cumprod, t, x_0.shape)
    sqrt_one_minus_alpha_cumprod_t = helper(sqrt_one_minus_alpha_cumprod, t, x_0.shape)
    
    # Based on parameterised formula for x[t], we add noise to the image.
    return sqrt_alpha_cumprod_t.to(device) * x_0.to(device) \
          + sqrt_one_minus_alpha_cumprod_t.to(device) * noise.to(device), noise.to(device)

