from torch import nn
import torch.nn.functional as F

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