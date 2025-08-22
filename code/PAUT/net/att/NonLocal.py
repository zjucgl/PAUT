import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        query = self.query_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        key = self.key_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        value = self.value_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        attention = torch.bmm(query.transpose(1, 2), key)  # Attention map
        attention = torch.softmax(attention, dim=-1)
        out = torch.bmm(value, attention)
        out = out.view(x.size(0), x.size(1), x.size(2), x.size(3))
        return self.out_conv(out)