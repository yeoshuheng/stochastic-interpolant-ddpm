import torch
import torch.nn.functional as F
from forward import beta, helper, sqrt_one_minus_alpha_cumprod, sqrt_recip_alpha, posterior_variance, forward_diffusion

@torch.no_grad()
def sample(x, t, model):
    """
    @param x: The noisy image.
    @param t: The timestep.
    @param model : The DDPM model.

    @return Predicted denoised image.
    """
    beta_t = helper(beta, t, x.shape)
    sqrt_one_minus_alpha_cumprod_t = helper(sqrt_one_minus_alpha_cumprod, t, x.shape)
    sqrt_recip_alpha_t = helper(sqrt_recip_alpha, t, x.shape)
    # Revert to denoised image (remove model's predicted noise)
    denoised = sqrt_recip_alpha_t * (x - beta_t * model(x, t) / sqrt_one_minus_alpha_cumprod_t)
    if t == 0:
        return denoised # At t == 0, we predict x_0 without noise.
    else:
        posterior_variance_t = helper(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return denoised + torch.sqrt(posterior_variance_t) * noise
    
def get_loss(x_0, t, model):
    """
    @param model : The DDPM model.
    @param x_0 : Original distribution.
    @param t : The timestep.
    """
    x_noised, noise = forward_diffusion(x_0, t)
    noise_pred = model(x_noised, t) # Takes in noised image at t to predict the noise.
    return F.l1_loss(noise, noise_pred)


