from torch import nn
import torch.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
# import torch.nn.utils.parametrizations.spectral_norm as spectral_norm

from normalization import SPADE

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.leaky_relu(x)
        return x


class GeneratorEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.leaky_relu(x)
        return x


# https://github.com/NVlabs/SPADE/blob/master/models/networks/architecture.py#L21
class SPADEResBlock(nn.Module):
    def __init__(self, fin, fout, segmap_channels):
        super().__init__()

        # self.learned_shortcut = (fin != fout)
        fmiddle = fout

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        self.conv_s = nn.Conv2d(fin, fout, kernel_size=3, padding=1)

        # apply spectral norm if specified
        # if 'spectral' in opt.norm_G:
        # if True:
        #     self.conv_0 = spectral_norm(self.conv_0)
        #     self.conv_1 = spectral_norm(self.conv_1)
        #     if self.learned_shortcut:
        #         self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(in_channels=fin, segmap_channels=segmap_channels)
        self.norm_1 = SPADE(in_channels=fmiddle, segmap_channels=segmap_channels)
        self.norm_s = SPADE(in_channels=fin, segmap_channels=segmap_channels)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        # if self.learned_shortcut:
        #     x_s = self.conv_s(self.actvn(self.norm_s(x, seg)))
        # else:
        #     x_s = x
        # return x_s
        return self.conv_s(self.actvn(self.norm_s(x, seg)))

    def actvn(self, x):
        # return F.leaky_relu(x, 2e-1)
        return F.relu(x)