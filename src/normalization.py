from torch import nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

def get_nonspade_norm_layer(norm_type='spectral-instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        subnorm_type = norm_type
        if norm_type.startswith('spectral-'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral-'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py#L66
class SPADE(nn.Module):
    def __init__(self, in_channels, segmap_channels):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(in_channels, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = segmap_channels // 4

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(segmap_channels, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, in_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, in_channels, kernel_size=3, padding=1)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out