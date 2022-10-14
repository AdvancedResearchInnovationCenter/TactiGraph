
import torch
from modules import MaxPooling, MaxPoolingX, ResidualBlock, spline_conv

class ResGNet(torch.nn.Module):
    
    def __init__(
        self, 
        n_classes, 
        layer_sizes=[64, 128, 256, 512],
        voxel_sizes=[20, 30, 50, 80]
        ):
        super(ResGNet, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.voxel_sizes = voxel_sizes

        self.n_graph_layers = len(layer_sizes)

        self.spline_conv_block1 = SplineConv(1, layer_sizes[0]) #Net.spline_block(1, 64)
        self.pool_block1 = MaxPooling(voxel_size=[0])

        self.res_blocks = torch.nn.ModuleDict({})
        self.pool_blocks = torch.nn.ModuleDict({})

        for layer in range(1, self.n_graph_layers):
            res_block = ResidualBlock(
                in_channel = self.layer_sizes[layer - 1], 
                out_channel = self.layer_sizes[layer]
                )

            self.res_blocks.update({f'res_block{layer+1}': res_block})

            if not layer+1 == self.n_graph_layers: #if not the last layer
                pool_block = MaxPooling(
                    voxel_size = self.voxel_sizes[layer], 
                    transform = T.Cartesian(cat=False)
                    )
            else:
                pool_block = MaxPoolingX(self.voxel_sizes[layer], 64)

            self.pool_blocks.update({f'pool_block{layer+1}': pool_block})

        self.fc1 = torch.nn.Linear(64 * layer_sizes[-1], 1024)
        self.fc2 = torch.nn.Linear(1024, n_classes)

    def forward(self, data):
        data = self.spline_conv_block1(data)
        data = self.pool_block1(data.x, data.pos, data.batch, data.edge_index)

        for layer in range(1, self.n_graph_layers):
            data = self.res_blocks[f'res_block{layer+1}'](data)
            if not layer+1 == self.n_graph_layers:
                data = self.pool_blocks[f'pool_block{layer+1}'](data.x, data.pos, data.batch, data.edge_index)
            else:
                x = self.pool_blocks[f'pool_block{layer+1}'](data.x, data.pos, data.batch)

        x = x.view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class spline_max_pool(torch.nn.Module):
    def __init__(
        self, 
        layer_sizes=[64, 128, 256, 512],
        voxel_sizes=[20, 30, 50, 80]
        ):
        super(spline_max_pool, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.voxel_sizes = voxel_sizes

        self.n_graph_layers = len(layer_sizes)

        self.spline_conv_blocks = torch.nn.ModuleDict({})
        self.pool_blocks = torch.nn.ModuleDict({})

        for layer in range(self.n_graph_layers):
            in_channel = self.layer_sizes[layer-1] if not layer == 0 else 1
            spline_conv_block = spline_conv(
                in_channels=in_channel,
                out_channels=self.layer_sizes[layer]
            )

            self.spline_conv_blocks.update({f'spline_conv_block{layer+1}': spline_conv_block})

            if not layer+1 == self.n_graph_layers:
                pool_block = MaxPooling(
                    voxel_size = self.voxel_sizes[layer], 
                    transform = T.Cartesian(cat=False)
                    )
            else:
                pool_block = MaxPoolingX(self.voxel_sizes[layer], 64)

            self.pool_blocks.update({f'pool_block{layer+1}': pool_block})

        self.fc1 = torch.nn.Linear(64 * layer_sizes[-1], 1024)
        self.fc2 = torch.nn.Linear(1024, 1)

    def forward(self, data):

        for layer in range(self.n_graph_layers):
            data = self.spline_conv_blocks[f'spline_conv_block{layer+1}'](data)
            if not layer+1 == self.n_graph_layers:
                data = self.pool_blocks[f'pool_block{layer+1}'](data.x, data.pos, data.batch, data.edge_index)
            else:
                x = self.pool_blocks[f'pool_block{layer+1}'](data.x, data.pos, data.batch)

        x = x.view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.elu(x, dim=1)