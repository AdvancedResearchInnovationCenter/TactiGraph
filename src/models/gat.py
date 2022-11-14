import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv, max_pool, max_pool_x, graclus, global_mean_pool, GCNConv,  global_mean_pool, SAGEConv, voxel_grid, SplineConv, GATv2Conv
import torch.nn.functional as F
import torch_scatter

class EventConv_mean_min_max_var(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(EventConv_mean_min_max_var, self).__init__()
        self.mlp = Seq(
            Linear(out_channels, out_channels), 
            ReLU(), 
            Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        sara = self.propagate(edge_index, x=x)
        return sara

    def aggregate(self, inputs, index):
        sums = torch_scatter.scatter_add(inputs, index, dim=0)
        maxs = torch_scatter.scatter_max(inputs, index, dim=0)[0]
        means = torch_scatter.scatter_mean(inputs, index, dim=0)
        var = torch.relu(
            torch_scatter.scatter_mean(
                inputs ** 2,
                index,
                dim=0) -
            means ** 2)

        aggrs = torch.hstack((sums, maxs, means, var))
        return self.mlp(aggrs)
    
    
class NetConnect_3e_model3(torch.nn.Module):
    def __init__(self):
        super(NetConnect_3e_model3, self).__init__()
        #self.conv1 = SplineConv(1, 64, dim=3, kernel_size=4)
        ####print("Kholous", len(data))

        in_dim = 3
        hidden_dim = in_dim*4
        self.edge_conv1 = EventConv_mean_min_max_var(in_dim, hidden_dim)

        in_dim2 = in_dim + hidden_dim
        hidden_dim2 = in_dim2 * 4
        ##print(in_dim2, hidden_dim2)
        self.edge_conv1_2L = EventConv_mean_min_max_var(in_dim2, hidden_dim2)

        in_dim3 = hidden_dim2 + in_dim2 
        hidden_dim3 = in_dim3 * 4 
        ##print(in_dim3, hidden_dim3)

        self.edge_conv1_3L = EventConv_mean_min_max_var(in_dim3, hidden_dim3)
        n_heads1 = 8
        self.conv0 = GATConv(hidden_dim3 + in_dim3, 64, n_heads1)
        self.bn0 = torch.nn.BatchNorm1d(64*n_heads1)

        self.conv1 = GATConv(64*n_heads1, 16, n_heads1)
        self.bn1 = torch.nn.BatchNorm1d(16*n_heads1)
        self.conv2 = GATConv(16*n_heads1, 16)
        self.bn2 = torch.nn.BatchNorm1d(16)
        
        self.fc1 = torch.nn.Linear(16 * 64, 128)
        self.fc2 = torch.nn.Linear(128, 2)


    def forward(self, data):

        source_data = data.x.clone().detach()
        data.x = self.edge_conv1(source_data, data.edge_index.clone().detach())
        data.x = torch.sigmoid(data.x)
        part1 = data.x
        ##print('part1', part1.shape)
        source_data_2L = torch.cat((source_data, part1), dim=1)
    
        ##print('source_data_2L', source_data_2L.shape)

        data.x = self.edge_conv1_2L(source_data_2L, data.edge_index.clone().detach())
        data.x = torch.sigmoid(data.x)
        part1_2L = data.x
        ##print('part1_2L', part1_2L.shape)
        source_data_3L = torch.cat((source_data_2L, part1_2L), dim=1)
        ##print('source_data_3L', source_data_3L.shape)
        data.x = self.edge_conv1_3L(
            source_data_3L, data.edge_index.clone().detach())
        data.x = torch.sigmoid(data.x)
        part1_3L = data.x
        ##print('part1_3L', part1_3L.shape)
        source_data_4L = torch.cat((source_data_3L, part1_3L), dim=1).to('cuda')
        ##print('source_data_4L', source_data_4L.shape)
                
        data.x = data.x.to('cuda')
        data.edge_index = data.edge_index.to('cuda')
        data.x = torch.sigmoid(self.conv0(source_data_4L, data.edge_index, data.edge_attr))
    
        data.x = self.bn0(data.x)
        data.x = torch.sigmoid(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn1(data.x)
        data.x = torch.sigmoid(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn2(data.x)
        
        cluster = voxel_grid(data.pos, batch=data.batch, size=0.25)
        x,_ = max_pool_x(cluster, data.x, batch=data.batch, size=64)
        
        x = x.view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.7)
        x = self.fc2(x)
        
        return x  # data.x