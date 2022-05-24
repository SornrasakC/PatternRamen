from torch import nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
# import torch.nn.utils.parametrizations.spectral_norm as spectral_norm
import torchvision

from src.normalization import SPADE
from normalization import get_nonspade_norm_layer

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, with_norm=True, spec_norm=True):
        super().__init__()
        if spec_norm:
            self.norm_layer = get_nonspade_norm_layer(norm_type='spectral-instance' if with_norm else 'spectral-none')
            self.conv = self.norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        else:
            self.norm = nn.InstanceNorm2d(out_channels) if with_norm else nn.Identity()
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),self.norm)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        return x


class GeneratorEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, with_norm=True, spec_norm=True):
        super().__init__()
        if spec_norm:
            self.norm_layer = get_nonspade_norm_layer(norm_type='spectral-instance' if with_norm else 'spectral-none')
            self.conv = self.norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        else:
            self.norm = nn.InstanceNorm2d(out_channels) if with_norm else nn.Identity()
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), self.norm)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        return x


# https://github.com/NVlabs/SPADE/blob/master/models/networks/architecture.py#L21
class SPADEResBlock(nn.Module):
    def __init__(self, fin, fout, segmap_channels, scale_factor=1, spec_norm=True):
        super().__init__()

        # self.learned_shortcut = (fin != fout)
        fmiddle = fout

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        self.conv_s = nn.Conv2d(fin, fout, kernel_size=3, padding=1)

        if spec_norm:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(in_channels=fin, segmap_channels=segmap_channels)
        self.norm_1 = SPADE(in_channels=fmiddle, segmap_channels=segmap_channels)
        self.norm_s = SPADE(in_channels=fin, segmap_channels=segmap_channels)

        self.up = nn.Upsample(scale_factor=scale_factor) if scale_factor != 1 else nn.Identity()

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        merged = x_s + dx

        out = self.up(merged)

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


# VGG architecture, used for the perceptual loss using a pretrained VGG network
class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        # self.slice4 = torch.nn.Sequential()
        # self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(12, 21):
        #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(21, 30):
        #     self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        self.criterion = nn.L1Loss()
        # self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8]

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        # h_relu4 = self.slice4(h_relu3)
        # h_relu5 = self.slice5(h_relu4)
        # out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        out = [h_relu1, h_relu2, h_relu3]
        return out
    

    def calc_p_loss(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        p_loss = 0
        for w, x_feat, y_feat in zip(self.weights, x_vgg, y_vgg): 
            p_loss += w * self.criterion(x_feat, y_feat.detach())
        return p_loss