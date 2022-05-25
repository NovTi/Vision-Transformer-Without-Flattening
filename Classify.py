import torch
import torch.nn as nn
from einops import rearrange


class Classify(nn.Module):
    def __init__(self, feature_size, cls_num, fp_num):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Linear(feature_size * fp_num, cls_num)

    def forward(self, x):
        b, n, c, h, w = x.shape
        results = torch.zeros((b, n, c, 1, 1))
        for i in range(b):
            results[i] = self.avgpool(x[i])
        results = rearrange(results, 'b n c h w -> b (n c h w)')
        results = self.mlp(results)
        return results