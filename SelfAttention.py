import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, OneHead, num_heads, in_channels, fp_num):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([])
        for i in range(num_heads):
            self.heads.append(OneHead(in_channels, fp_num))

        self.conv = nn.Conv2d(num_heads * in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # get different results from different heads
        b, _, _, _, _ = x.shape
        return_x = x.clone()
        results = self.heads[0](x)

        for i in range(1, self.num_heads):
            wait_cat = self.heads[i](x)
            results = torch.cat((results, wait_cat), dim=2)

        for i in range(b):
            return_x[i] = self.conv(results[i])
        return return_x
