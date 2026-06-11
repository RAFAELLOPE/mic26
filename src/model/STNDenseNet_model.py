from torch import nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121


class STN_DenseNet(nn.Module):
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
        super(STN_DenseNet, self).__init__()
        self.dense_net = DenseNetModel(
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

        # Spatial transformer Localization network
        self.localization_net = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=7),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True),
            nn.Conv3d(8, 10, kernel_size=2),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Regressor for the 3 * 4 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10*30*30*30, 32),
            nn.ReLU(True),
            nn.Linear(32, 3*4)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor(
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], 
                dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization_net(x)
        xs = xs.view(-1, 10*30*30*30)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 4)

        grid = F.affine_grid(
            theta, 
            x.size(), 
            align_corners=True
        )
        x = F.grid_sample(
            x, 
            grid, 
            align_corners=True
        )

        return x
        
    def forward(self, x):
        x = self.stn(x)
        x = self.dense_net(x)
        return x