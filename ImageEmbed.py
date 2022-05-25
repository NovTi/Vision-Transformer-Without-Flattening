import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange


class ImageEmbed(nn.Module):
    def __init__(self, img_channels, out_channels):
        super().__init__()
        self.image_embed = nn.Sequential(
            # extract feature
            nn.Conv2d(img_channels, out_channels, kernel_size=9, stride=1),
            nn.ReLU(),
            Rearrange('b (c n) h w -> b n c h w', c=4),
        )
        # class embedding
        self.cls_token = nn.Parameter(torch.rand((1, 4, 120, 120)))

    def forward(self, x):
        b, _, _, _ = x.shape
        out = self.image_embed(x)
        # concat class token
        cls_tokens = repeat(self.cls_token, 'n c h w -> b n c h w', b=b)
        out = torch.cat([cls_tokens, out], dim=1)

        return out