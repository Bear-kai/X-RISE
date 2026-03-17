import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layers.torch import Rearrange


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)   # out_size = (in_size + 2*p - k) // s + 1, no dilation

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)                  # out_size = s*(in_size-1) + k - 2*p
        self.conv = nn.ConvTranspose1d(dim, dim, 3, 2, 1, output_padding=1)  # change! out_size = s*(in_size-1) + k - 2*p + out_padding
                                                                             
    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


def test():
    cb = Conv1dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1,256,16))     # (B,D,L)
    o = cb(x)
