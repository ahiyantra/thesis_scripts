# "wgan_models_v2A1.py" ~ "v2.151"
# Model definitions for WGAN-SN (Spectral Normalization).
# Contains Generator, Critic, and weight initialization.
# Based on v2A with same architecture but updated for consistency with v2.151.

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    """Generator Network for WGAN-SN (DCGAN-style architecture)."""
    def __init__(self, noise_dim, channels_img, features_g):
        super(Generator, self).__init__()
        # Input: N x noise_dim x 1 x 1
        self.net = nn.Sequential(
            # Z -> FEATURES_G*16 x 4 x 4
            self._block(noise_dim, features_g * 16, 4, 1, 0), 
            # -> FEATURES_G*8 x 8 x 8
            self._block(features_g * 16, features_g * 8, 4, 2, 1), 
            # -> FEATURES_G*4 x 16 x 16
            self._block(features_g * 8, features_g * 4, 4, 2, 1), 
            # -> FEATURES_G*2 x 32 x 32
            self._block(features_g * 4, features_g * 2, 4, 2, 1), 
            # -> FEATURES_G x 64 x 64
            self._block(features_g * 2, features_g, 4, 2, 1), 
            # -> CHANNELS_IMG x 128 x 128
            nn.ConvTranspose2d(features_g, channels_img, kernel_size=4, stride=2, padding=1), 
            # Output image in [-1, 1] range
            nn.Tanh() 
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """Helper for creating a Generator block (ConvTranspose2d + BatchNorm + ReLU)."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(True),
        )

    def forward(self, x):
        """Forward pass through the Generator."""
        return self.net(x)


class CriticSN(nn.Module):
    """Critic Network for WGAN-SN with Spectral Normalization for Lipschitz constraint."""
    def __init__(self, channels_img, features_c):
        super(CriticSN, self).__init__()
        # Input: N x channels_img x 128 x 128
        self.net = nn.Sequential(
            # Layer 1: 128 -> 64
            spectral_norm(nn.Conv2d(channels_img, features_c, kernel_size=4, stride=2, padding=1)), 
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2: 64 -> 32
            spectral_norm(nn.Conv2d(features_c, features_c * 2, kernel_size=4, stride=2, padding=1)), 
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3: 32 -> 16
            spectral_norm(nn.Conv2d(features_c * 2, features_c * 4, kernel_size=4, stride=2, padding=1)), 
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4: 16 -> 8
            spectral_norm(nn.Conv2d(features_c * 4, features_c * 8, kernel_size=4, stride=2, padding=1)), 
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 5: 8 -> 4
            spectral_norm(nn.Conv2d(features_c * 8, features_c * 16, kernel_size=4, stride=2, padding=1)), 
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 6: 4 -> 1x1 scalar output
            spectral_norm(nn.Conv2d(features_c * 16, 1, kernel_size=4, stride=1, padding=0)), 
        )

    def forward(self, x):
        """Forward pass through the Critic."""
        return self.net(x)


def initialize_weights(model):
    """Initializes weights according to DCGAN paper recommendations."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d): 
             # Initialize BatchNorm weights to N(1, 0.02) and biases to 0
             if m.weight is not None: 
                 nn.init.normal_(m.weight.data, 1.0, 0.02)
             if m.bias is not None: 
                 nn.init.constant_(m.bias.data, 0)
