import torch
import torch.nn as nn
from einops import rearrange


class ConvBlock(nn.Module):
    # fp_num is feature patch number
    def __init__(self, in_channels, fp_num, stride, c_expand):
        super().__init__()
        self.batchnorm = nn.BatchNorm3d(fp_num)
        self.conv_series = nn.ModuleList([])
        self.stride = stride
        self.c_expand = c_expand
        for i in range(fp_num):
            self.conv_series.append(nn.Sequential(
                nn.Conv2d(in_channels, c_expand * in_channels, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(c_expand * in_channels, c_expand * in_channels, kernel_size=3, stride=stride, padding=1),
                nn.ReLU(),
                nn.Conv2d(c_expand * in_channels, c_expand * in_channels, kernel_size=1, stride=1)
            ))

    def forward(self, x):
        b, n, c, h, w = x.shape
        if self.stride == 2:
            results = torch.zeros((b, n, self.c_expand * c, (h + 1) // 2, (w + 1) // 2))
        else:
            results = torch.zeros((b, n, self.c_expand * c, h, w))

        results = rearrange(results, 'b n c h w -> n b c h w')

        x = self.batchnorm(x)
        # change the shape for efficient convolution
        # feature patches with the same position have the same conv series
        x = rearrange(x, 'b n c h w -> n b c h w')
        for i in range(n):
            results[i] = self.conv_series[i](x[i])
        results = rearrange(results, 'n b c h w -> b n c h w')

        return results
