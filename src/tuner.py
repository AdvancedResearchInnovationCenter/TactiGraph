import torch

from torch.nn import Linear
from torch.nn.functional import elu, dropout
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian

from ray import tune
from ray.tune import CLIReporter

from imports.TactileDataset import TactileDataset
from imports.TrainModel import TrainModel
from models.modules import MaxPooling, MaxPoolingX

import numpy as np

class model2(torch.nn.Module):

    def __init__(self, pooling_size=(16/346, 12/260), pooling_outputs=32, pooling_after_conv2=False, more_layer=False, more_block=False):
        super(model2, self).__init__()
        #n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
             #0, 1, 2 , 3 , 4 , 5 , 6 , 7 , 8
        #kernel_size = 8
        #n = [1, 16, 32, 32, 32, 128, 128, 128]
        #pooling_outputs = 128
        dim = 3

        bias = False
        root_weight = False

        # Set dataset specific hyper-parameters.
        kernel_size = 2

        self.pooling_size = np.array(pooling_size)
        self.pooling_output = pooling_outputs
        self.pooling_after_conv2 = pooling_after_conv2
        self.more_layer = more_layer
        self.more_block = more_block

        self.conv1 = SplineConv(1, 8, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm1 = BatchNorm(in_channels=8)

        if self.more_layer:
            self.conv1_1 = SplineConv(8, 8, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm1_1 = BatchNorm(in_channels=8)

        self.conv2 = SplineConv(8, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm2 = BatchNorm(in_channels=16)

        self.conv2_1 = SplineConv(16, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm2_1 = BatchNorm(in_channels=16)
        self.pool2_1 = MaxPooling(self.pooling_size/2, transform=Cartesian(norm=True, cat=False))

        self.conv3 = SplineConv(16, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm3 = BatchNorm(in_channels=16)

        self.conv4 = SplineConv(16, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm4 = BatchNorm(in_channels=16)

        self.conv5 = SplineConv(32, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm5 = BatchNorm(in_channels=32)
        self.pool5 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

        self.conv6 = SplineConv(pooling_outputs, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm6 = BatchNorm(in_channels=pooling_outputs)
        self.conv7 = SplineConv(pooling_outputs, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm7 = BatchNorm(in_channels=pooling_outputs)

        if self.more_block:
            self.conv6_1 = SplineConv(pooling_outputs, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm6_1 = BatchNorm(in_channels=pooling_outputs)
            self.conv7_1 = SplineConv(pooling_outputs, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm7_1 = BatchNorm(in_channels=pooling_outputs)


        self.pool_final = MaxPoolingX(0.25, size=16)
        self.fc = Linear(pooling_outputs * 16, out_features=2, bias=bias)

    def forward(self, data):
        data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm1(data.x)

        if self.more_layer:
            data.x = elu(self.conv1_1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm1_1(data.x)
        
        data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2(data.x)
        
        data.x = elu(self.conv2_1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2_1(data.x)

        if self.pooling_after_conv2:
            data = self.pool2_1(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)
        
        x_sc = data.x.clone()
        data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm3(data.x)
        data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm4(data.x)
        data.x = data.x + x_sc

        data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm5(data.x)
        data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        x_sc = data.x.clone()
        data.x = elu(self.conv6(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm6(data.x)
        data.x = elu(self.conv7(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm7(data.x)
        data.x = data.x + x_sc

        if self.more_block:
            data.x = elu(self.conv6_1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm6_1(data.x)
            data.x = elu(self.conv7_1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm7_1(data.x)
            data.x = data.x + x_sc


        x = self.pool_final(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)

model = model2(pooling_size=(16/346, 12/260), pooling_outputs=64, pooling_after_conv2=False, more_layer=False, more_block=False)
tm = TrainModel(
    '/media/hussain/drive1/tactile-data/extractions/morethan3500ev_lessthan_9deg', 
    model, 
    lr=0.001, 
    features='pol', 
    batch=6, 
    n_epochs=1000, 
    experiment_name='test', 
    desc='/media/hussain/drive1/tactile-data/extractions/morethan3500ev_lessthan_9deg',
    merge_test_val=False,
    loss_func=torch.nn.MSELoss()
    
)
def train_cifar(config, checkpoint_dir=None, data_dir=None):
    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")