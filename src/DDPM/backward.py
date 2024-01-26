import torch.nn as nn

# Backward Process
# U-Net architecture (down-sample -> bottleneck -> up-sample), input becomes smaller
# but becomes deeper as more channels are added throughout the process.
# U-Net also fufills dimensional requirements of input and output.
# This model learns mean of gaussian of the noise. (Score-matching)
# During training, we randomly select timesteps to train from, however, during inference,
# we need to sample through the whole process.

# Timestep encoding needed to inform the model about the timestep it is in.

class UNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_embedding_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_embedding_dim, out_channel)
        if up: # Selection between up-sample and down-sample block.
            # We multiply by 2 to handle the incoming residuals for up-sampling.
            self.conv1 = nn.Conv2d(2 * in_channel, out_channel, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
            self.transform = nn.Conv2d(out_channel, out_channel, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride = 2)
        self.bnorm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU() # Used in CNN to prevent exponential growth in computational requirements.
        
    def forward(self, x, t):
        first_conv = self.bnorm(self.relu(self.conv1(x)))
        time_embedding = self.relu(self.time_mlp(t))
        # Extend time embedding dimensions to match current number of channels.
        time_embedding = time_embedding[(..., ) + (None, ) * 2]
        time_embedded_conv = time_embedding + first_conv
        second_conv = self.bnorm(self.relu(self.conv2(time_embedded_conv)))
        return self.transform(second_conv)


