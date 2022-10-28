import torch
from torch_geometric.nn import SplineConv, voxel_grid, max_pool, max_pool_x
import torch.nn.functional as F
import torch_geometric.transforms as T
class nvs_no_skip(torch.nn.Module):
    def __init__(self):
        super(nvs_no_skip, self).__init__()
        self.conv1 = SplineConv(1, 64, dim=3, kernel_size=3)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = SplineConv(64, 128, dim=3, kernel_size=3)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.conv3 = SplineConv(128, 256, dim=3, kernel_size=3)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.conv4 = SplineConv(256, 512, dim=3, kernel_size=3)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.fc1 = torch.nn.Linear(64 * 512, 1024)
        self.fc2 = torch.nn.Linear(1024, 2)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn1(data.x)
        cluster = voxel_grid(data.pos, batch=data.batch, size=0.05)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn2(data.x)
        cluster = voxel_grid(data.pos, batch=data.batch, size=0.1)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        
        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn3(data.x)
        cluster = voxel_grid(data.pos,batch= data.batch, size=0.15)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn4(data.x)
        cluster = voxel_grid(data.pos, batch=data.batch, size=0.25)
        x,_ = max_pool_x(cluster, data.x, batch=data.batch, size=64)

        x = x.view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return x
