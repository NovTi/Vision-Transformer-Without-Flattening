import torch
import torch.nn as nn
from einops import rearrange


class ViTEncoder(nn.Module):
    def __init__(self, SA, CB, SAHead, num_heads, in_channels,
                 fp_num, stride, c_expand):
        super().__init__()
        self.SelfAtt = SA(SAHead, num_heads, in_channels, fp_num)
        self.ConvBlock = CB(in_channels, fp_num, stride, c_expand)
        self.stride = stride
        self.c_expand = c_expand
        self.down_sample_convs = nn.ModuleList([])
        for i in range(fp_num):
            self.down_sample_convs.append(
                nn.Conv2d(in_channels, c_expand * in_channels, kernel_size=3, stride=stride, padding=1),
            )
        self.batchnorm = nn.BatchNorm3d(fp_num)

    def forward(self, x):
        b, n, c, h, w = x.shape
        # first resdual connection
        identity = x
        x = self.SelfAtt(x)
        x = x + identity
        # second resudial connection
        identity = x
        # change the size of identity
        if self.stride == 2:
            ident = torch.zeros((n, b, self.c_expand * c, (h + 1) // 2, (w + 1) // 2))
            identity = self.batchnorm(identity)
            identity = rearrange(identity, 'b n c h w -> n b c h w')
            for i in range(n):
                ident[i] = self.down_sample_convs[i](identity[i])
            identity = rearrange(ident, 'n b c h w -> b n c h w')
        else:
            ident = torch.zeros((n, b, self.c_expand * c, h, w))
            identity = self.batchnorm(identity)
            identity = rearrange(identity, 'b n c h w -> n b c h w')
            for i in range(n):
                ident[i] = self.down_sample_convs[i](identity[i])
            identity = rearrange(ident, 'n b c h w -> b n c h w')

        x = self.ConvBlock(x)
        x = x + identity

        return x