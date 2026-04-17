import torch
import torch.nn as nn
from typing import Tuple, Optional
from einops import rearrange

class Reshape(nn.Module):
    """Inline reshape: (batch, C*H*W) -> (batch, C, H, W)."""
    def __init__(self, C: int, H: int, W: int):
        super().__init__()
        self.C, self.H, self.W = C, H, W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x
        return rearrange(x, 'b (c h w) -> b c h w', c=self.C, h=self.H, w=self.W)

class NatureCNN(nn.Module):
    """
    Nature-DQN conv stack for image-based observations.
    """
    def __init__(self, C: int, H: int, W: int):
        super().__init__()
        self.net = nn.Sequential(
            Reshape(C, H, W),
            nn.Conv2d(C, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.output_dim = self._get_output_dim(C, H, W)

    def _get_output_dim(self, C, H, W) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            return self.net(dummy).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
