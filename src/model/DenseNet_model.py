import torch
from torch import nn
from monai.networks.nets import DenseNet121

class DenseNetModel(nn.Module):
    def __init__(
            self,
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            init_features=64,
            growth_rate=32,
            block_config=(6, 12, 24, 16), 
            bn_size=4,
            act='relu',
            norm='batch',
            dropout_prob=0.0
    ):
        default_dense_net = DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config, 
            bn_size=bn_size,
            act=act,
            norm=norm,
            dropout_prob=dropout_prob
        )

        self.model = default_dense_net

    def forward(self, x):
        return self.model(x)