import torch
import torch.nn as nn
from einops import rearrange


class SAHead1(nn.Module):
    def __init__(self, in_channels, fp_num):
        super().__init__()

        self.q_convs = nn.ModuleList([])

        # set the value kernels
        self.v_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.v_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # set the key kernels
        self.k_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=20, stride=5)
        self.k_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2)

        # set the query kernels
        for i in range(fp_num):
            self.q_convs.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2),
                nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=1)
            ])

        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        return_x = x

        # b - batch size  n: feature patch numbers
        b, n, _, _, _ = x.shape
        for i in range(b):
            values = self.v_conv1(x[i])
            values = self.v_conv2(values)

            keys = self.k_conv1(x[i])
            keys = self.k_conv2(keys)

            for j in range(n):
                querys = self.q_convs[2 * j](keys)
                querys = self.q_convs[2 * j + 1](querys)

                querys = rearrange(querys, 'n c h w -> n (c h w)')

                softmax = self.softmax(querys)
                querys = rearrange(softmax, 'n c -> n c 1 1')

                return_x[i][j] = torch.sum(querys * values, dim=0)

        return return_x