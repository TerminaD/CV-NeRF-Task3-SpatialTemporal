import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    The neural network for NeRF.
    
    This class is ported from kwea123/nerf_pl 
    (https://github.com/kwea123/nerf_pl/tree/master)
    under the MIT license.
    """
    def __init__(self,
                 D=8, 
                 W=256,
                 in_channels_xyz=60, 
                 in_channels_dir=24, 
                 in_channels_time=8,
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3*10*2=60 by default)
        in_channels_dir: number of input channels for direction (3*4*2=24 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_time = in_channels_time
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz+in_channels_time, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz+in_channels_time, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Sequential(nn.Linear(W, 1),
                                   nn.Sigmoid())
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())
        

    def forward(self, x):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        input_xyz, input_dir, input_time = \
        torch.split(x, [self.in_channels_xyz, self.in_channels_dir, self.in_channels_time], dim=-1)

        xyz_ = torch.concat([input_xyz, input_time], -1)
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, input_time, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out
    