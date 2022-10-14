import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class ya_1(torch.nn.Module):
    def __init__(self):
        super(ya_1, self).__init__()

        self.conv1 = GCNConv(1, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)

        self.conv2 = GCNConv(64, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        
        self.conv3 = GCNConv(128, 512)
        self.bn3 = torch.nn.BatchNorm1d(512) 
        
        self.conv4 = GCNConv(512, 256)
        self.bn4 = torch.nn.BatchNorm1d(256)  
        
        self.conv5 = GCNConv(256, 64)
        self.bn5 = torch.nn.BatchNorm1d(64)
        
        self.conv6 = GCNConv(64, 32)
        self.bn6 = torch.nn.BatchNorm1d(32)

        self.conv7 = GCNConv(32, 16)
        self.bn7 = torch.nn.BatchNorm1d(16)        

        self.conv8 = GCNConv(16, 2)
        self.bn8 = torch.nn.BatchNorm1d(16)    
        

        self.fc1 = torch.nn.Linear(16, 16)
        self.fc2 = torch.nn.Linear(16, 2)
        


    def forward(self, data):
        #data.x = F.elu(self.conv1(data.x , data.edge_index , data.edge_attr ))
        # print("index", data.x.view(-1,2).shape)
        data.x = F.sigmoid(self.conv1(torch.tensor(data.x.view(-1,1), dtype=torch.float32), data.edge_index) )#, data.edge_attr ))

        # data.x = F.sigmoid(self.conv1(torch.tensor(data.x.clone().detach(), dtype=torch.float32), data.edge_index))
        data.x = self.bn1(data.x)
        part1= data.x


        data.x = F.sigmoid(self.conv2(data.x, data.edge_index))
        data.x = self.bn2(data.x)
        part2= data.x

        
        data.x=torch.cat((part1,part2), dim=1) #64+64
        data.x = F.sigmoid(self.conv3(data.x, data.edge_index))
        data.x = self.bn3(data.x)
        
        data.x = F.sigmoid(self.conv4(data.x, data.edge_index))
        data.x = self.bn4(data.x)
        
        data.x = F.sigmoid(self.conv5(data.x, data.edge_index))
        data.x = self.bn5(data.x)

        data.x = F.sigmoid(self.conv6(data.x, data.edge_index))
        data.x = self.bn6(data.x)
        
        data.x = F.sigmoid(self.conv7(data.x, data.edge_index))
        data.x = self.bn7(data.x)

        # data.x = F.sigmoid(self.conv8(data.x, data.edge_index))
                # 2. Readout layer
        x = global_mean_pool(data.x, data.batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.fc1(x))
        #print("abdulrahman2 done",x.size())

        out = F.elu(self.fc2(x))
        
        # print("out", out)
#         out=F.softmax(data.x, dim=1)
        return out

