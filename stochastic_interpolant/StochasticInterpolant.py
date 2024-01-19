import torch.nn as nn
import torch

class StochasticInterpolant(nn.Module):
    def __init__(self, params : dict, models : dict, sde_type="vs"):
        self.params = params
        self.n_net = models['n_net']
        self.b_net = models['b_net']
        self.v_net = models['v_net']
        self.s_net = models['s_net']
        self.d = 0.3
        self.sde_type = sde_type
        self.gamma_inv_max = 10
        self.t_min = 0.01

    """
    Gamma & Epsilon
    """

    @torch.no_grad()
    def epsilon(self, t):
        return t * (1 - t)
    
    @torch.no_grad()
    def gamma(self, t):
        return 1.4142 * torch.sqrt(t * (1 - t))

    @torch.no_grad()
    def gamma_der(self, t):
        return (1 - 2 * t) / torch.sqrt(2 * (1 - torch.pow(t, 2)) + 1e-4)
    
    @torch.no_grad()
    def gamma_inv(self, t):
        return torch.clamp(
            1 / (1.4142 * torch.sqrt(t * (1 - t) + 1e-4)), 0.0, self.gamma_inv_max
        )
    
    """
    Loss Functions
    """

    def v_loss(self, v_net, t, x_t, x_0, x_1):
        t = torch.clip(t, self.t_min, 1 - self.t_min)
        d_t = x_1 - x_0
        pred_v = v_net(x_t, t)
        total_loss =  0.5 * torch.norm(pred_v) ** 2 - d_t * pred_v
        return torch.mean(total_loss)
    
    def b_loss(self, b_net, t, x_t, x_0, x_1, z):
        t = torch.clip(t, self.t_min, 1 - self.t_min)
        d_t = x_1 - x_0
        pred_b = b_net(x_t, t)
        total_loss = 0.5 * torch.norm(pred_b) ** 2 - (d_t + self.gamma(t) * z) * pred_b
        return torch.mean(total_loss)

    def s_loss(self, s_net, t, x_t, x_0, x_1, z):
        t = torch.clip(t, self.t_min, 1 - self.t_min)
        pred_s = s_net(x_t, t)
        total_loss = 0.5 * torch.norm(pred_s) ** 2 - self.gamma_inv(t) * z * pred_s
        return torch.mean(total_loss)
    
    def n_loss(self, n_net, t, x_t, x_0, x_1, z):
        t = torch.clip(t, self.t_min, 1 - self.t_min)
        pred_n = n_net(x_t, t)
        total_loss = 0.5 * torch.norm(pred_n) ** 2 - z * pred_n 
        return torch.mean(total_loss)
    
    def total_loss(self, x_0, x_1):
        # generate random timesteps
        t = torch.randint(low=0, high=1, size=(32,))
        x_t = self.sample_xt(t, x_0, x_1)
        match self.sde_type:
            case "vs":
                loss_1 = self.v_loss(self.v_net, t, x_t, x_0, x_1)
                loss_2 = self.s_loss(self.s_net, t, x_t, x_0, x_1)
            case "bs":
                loss_1 = self.b_loss(self.b_net, t, x_t, x_0, x_1)
                loss_2 = self.s_loss(self.s_net, t, x_t, x_0, x_1)
            case "bn":
                loss_1 = self.b_loss(self.b_net, t, x_t, x_0, x_1)
                loss_2 = self.n_loss(self.s_net, t, x_t, x_0, x_1)
            case "vn":
                loss_1 = self.v_loss(self.v_net, t, x_t, x_0, x_t)
                loss_2 = self.n_loss(self.n_net, t, x_t, x_0, x_1)
        return loss_1 + loss_2

    """
    Sampling
    """

    def sample_xt(self, t, x_0, x_1):
        t  = torch.clip(t, self.t_min, t - self.t_min)
        # generate noise
        z = torch.randn_like(x_0).float()
        x_t = (1 - t) * x_0 + t * x_1 + self.gamma(t) * z
        return x_t, z
    
    """
    Stochastic Differentials

        - Regenerates the optimal action x_t from x_0 by sampling 
    """

    @torch.no_grad()
    def sde_bs(self, x_0, direction="forward"):
        n_timesteps = self.params["n_timesteps"]
        samples = [[]] * (n_timesteps + 1)
        samples[0] = x_0
        delta_t = torch.full((32,), 1 / n_timesteps, dtype=torch.float32)
        for step in range(1, n_timesteps+1):
            x_curr = samples[step - 1]
            t_tensor = torch.full((32,), step / n_timesteps, dtype=torch.float32)
            t_tensor = torch.clip(t_tensor, self.t_min, 1 - self.t_min)
            dW = self.d * torch.randn_like(x_curr, dtype=torch.float32)
            match direction:
                case "forward":
                    b = self.b_net(x_curr, t_tensor)
                    s = self.s_net(x_curr, t_tensor)
                    eps = self.epsilon(t_tensor)
                    dX = delta_t * (b + s * eps) + torch.sqrt(2 * eps) * dW
                case "backward":
                    b = self.b_net(x_curr, 1 - t_tensor)
                    s = self.s_net(x_curr, 1 - t_tensor)
                    eps = self.epsilon(1 - t_tensor)
                    dX = delta_t * (b - s * eps) + torch.sqrt(2 * eps) * dW
            x_next = x_curr + dX
            samples[step] = x_next
        return samples[-1], samples

    @torch.no_grad()
    def sde_bn(self, x_0, direction="forward"):
        n_timesteps = self.params["n_timesteps"]
        samples = [[]] * (n_timesteps + 1)
        samples[0] = x_0
        delta_t = torch.full((32,), 1 / n_timesteps, dtype=torch.float32)
        for step in range(1, n_timesteps+1):
            x_curr = samples[step - 1]
            t_tensor = torch.full((32,), step / n_timesteps, dtype=torch.float32)
            t_tensor = torch.clip(t_tensor, self.t_min, 1 - self.t_min)
            dW = self.d * torch.randn_like(x_curr, dtype=torch.float32)
            match direction:
                case "forward":
                    b = self.b_net(x_curr, t_tensor)
                    denoise = self.n_net(x_curr, t_tensor)
                    denoiser_factor = -self.gamma_inv(t_tensor)
                    s = torch.multiply(denoise, denoiser_factor)
                    eps = self.epsilon(t_tensor)
                    dX = delta_t * (b + s * eps) + torch.sqrt(2 * eps) * dW
                case "backward":
                    b = self.b_net(x_curr, 1 - t_tensor)
                    denoise = self.n_net(x_curr, 1 - t_tensor)
                    denoiser_factor = -self.gamma_inv(1 - t_tensor)
                    s = torch.multiply(denoise, denoiser_factor)
                    eps = self.epsilon(1 - t_tensor)
                    dX = delta_t * (b - s * eps) + torch.sqrt(2 * eps) * dW
            x_next = x_curr + dX
            samples[step] = x_next
        return samples[-1], samples
    
    @torch.no_grad()
    def sde_vn(self, x_0, direction="forward"):
        n_timesteps = self.params["n_timesteps"]
        samples = [[]] * (n_timesteps + 1)
        samples[0] = x_0
        delta_t = torch.full((32,), 1 / n_timesteps, dtype=torch.float32)
        for step in range(1, n_timesteps+1):
            x_curr = samples[step - 1]
            t_tensor = torch.full((32,), step / n_timesteps, dtype=torch.float32)
            t_tensor = torch.clip(x_curr, self.t_min, 1 - self.t_min)
            dW = self.d * torch.randn_like(t_tensor, dtype=torch.float32)
            match direction:
                case "forward":
                    v = self.v_net(x_curr, t_tensor)
                    denoiser = self.n_net(x_curr, t_tensor)
                    gamma_gamma = self.gamma(t_tensor) * self.gamma_der(t_tensor)
                    s = torch.multiply(denoiser, denoiser_factor)
                    s *= gamma_gamma
                    b = v - gamma_gamma * s
                    eps = self.epsilon(1 - t_tensor)
                    dX = delta_t * (b - s * eps) + torch.sqrt(2 * eps) * dW
                case "backward":
                    v = self.v_net(x_curr, 1 - t_tensor)
                    denoiser = self.n_net(x_curr, 1 - t_tensor)
                    gamma_gamma = self.gamma(1 - t_tensor) * self.gamma_der(1 - t_tensor)
                    denoiser_factor = -self.gamma_inv(1 - t_tensor)
                    s = torch.multiply(denoiser, denoiser_factor)
                    b = v - gamma_gamma * s
                    eps = self.epsilon(1 - t_tensor)
                    dX = delta_t * (b - s * eps) + torch.sqrt(2 * eps) * dW
            x_net = x_curr + dX
            samples[step] = x_net
        return samples[-1], samples

    @torch.no_grad()
    def sde_vs(self, x_0, direction="forward"):
        n_timesteps = self.params["n_timesteps"]
        samples = [[]] * (n_timesteps + 1)
        samples[0] = x_0
        delta_t = torch.full((32,), 1 / n_timesteps, dtype=torch.float32)
        for step in range(1, n_timesteps+1):
            x_curr = samples[step - 1]
            t_tensor = torch.full((32,), step / n_timesteps, dtype=torch.float32)
            t_tensor = torch.clip(x_curr, self.t_min, 1 - self.t_min)
            dW = self.d * torch.randn_like(t_tensor, dtype=torch.float32)
            match direction:
                case "forward":
                    v = self.v_net(x_curr, t_tensor)
                    s = self.s_net(x_curr, t_tensor)
                    gamma_gamma = self.gamma(t_tensor) * self.gamma_der(t_tensor)
                    b = v - gamma_gamma * s
                    eps = self.epsilon(t_tensor)
                    dX = delta_t * (b + s * eps) + torch.sqrt(2 * eps) * dW
                case "backward":
                    v = self.v_net(x_curr, 1 - t_tensor)
                    gamma_gamma = self.gamma(1 - t_tensor) * self.gamma_der(1 - t_tensor)
                    b = v  - gamma_gamma * s
                    s = self.s_net(x_curr, 1 - t_tensor)
                    eps = self.epsilon(1 - t_tensor)
                    dX = delta_t * (b - s * eps) + torch.sqrt(2 * eps) * dW
            x_next = x_curr + dX
            samples[step] = x_next
        return samples[-1], samples
    
