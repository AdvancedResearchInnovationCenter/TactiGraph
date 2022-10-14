import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, voxel_grid, max_pool, max_pool_x
from torch_geometric.data import Data


transform = T.Cartesian(cat=False)

class spline_conv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, dim=3, kernel_size=5):
        super(spline_conv, self).__init__()

        self.spline_conv = SplineConv(in_channels, out_channels, dim=dim, kernel_size=kernel_size)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, data):
        spline_convd = self.spline_conv(data.x, data.edge_index, data.edge_attr)
        data.x = F.elu(spline_convd)
        data.x = self.bn(data.x)
        return data

class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.left_spline_conv1 = SplineConv(in_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn1 = torch.nn.BatchNorm1d(out_channel)
        self.left_spline_conv2 = SplineConv(out_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn2 = torch.nn.BatchNorm1d(out_channel)
        
        self.shortcut_spline_conv = SplineConv(in_channel, out_channel, dim=3, kernel_size=1)
        self.shortcut_bn = torch.nn.BatchNorm1d(out_channel)
        
     
    def forward(self, data):
        data.x = F.elu(self.left_bn2(self.left_spline_conv2(F.elu(self.left_bn1(self.left_spline_conv1(data.x, data.edge_index, data.edge_attr))),
                                            data.edge_index, data.edge_attr)) + 
                       self.shortcut_bn(self.shortcut_spline_conv(data.x, data.edge_index, data.edge_attr)))
        
        print('res forward', data)
        return data


#https://github.com/uzh-rpg/aegnn/blob/master/aegnn/models/layer/max_pool.py
class MaxPooling(torch.nn.Module):

    def __init__(self, voxel_size, transform = None):
        super(MaxPooling, self).__init__()
        self.voxel_size = voxel_size
        self.transform = transform

    def forward(self, x, pos, batch = None, edge_index = None, return_data_obj = True):
        assert edge_index is not None, "edge_index must not be None"

        cluster = voxel_grid(pos, batch=batch, size=self.voxel_size)
        data = Data(x=x, pos=pos, edge_index=edge_index, batch=batch).to('cuda')
        data = max_pool(cluster, data=data, transform=self.transform)  # transform for new edge attributes
        if return_data_obj:
            return data
        else:
            return data.x, data.pos, getattr(data, "batch"), data.edge_index, data.edge_attr

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size})"


#https://github.com/uzh-rpg/aegnn/blob/master/aegnn/models/layer/max_pool_x.py
class MaxPoolingX(torch.nn.Module):

    def __init__(self, voxel_size, size: int):
        super(MaxPoolingX, self).__init__()
        self.voxel_size = voxel_size
        self.size = size

    def forward(self, x, pos, batch = None):
        cluster = voxel_grid(pos, batch=batch, size=self.voxel_size)
        x, _ = max_pool_x(cluster, x, batch, size=self.size)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size}, size={self.size})"
