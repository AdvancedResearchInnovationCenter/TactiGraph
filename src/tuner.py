import torch

from torch.nn import Linear
from torch.nn.functional import elu, dropout
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from imports.TactileDataset import TactileDataset
from imports.TrainModel import TrainModel
from models.modules import MaxPooling, MaxPoolingX
from tqdm.auto import trange, tqdm
import numpy as np
from math import pi
import os
from functools import partial

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

        self.conv5 = SplineConv(16, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm5 = BatchNorm(in_channels=pooling_outputs)
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


def train_cifar(config, checkpoint_dir=None, data_dir=None):
    net = model2(**config)
    print(net)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    net.to(device)

    tm = TrainModel(
        '/media/hussain/drive1/tactile-data/extractions/morethan3500ev_lessthan_9deg', 
        net, 
        lr=0.001, 
        features='pol', 
        batch=6, 
        n_epochs=1000, 
        experiment_name='test', 
        desc='/media/hussain/drive1/tactile-data/extractions/morethan3500ev_lessthan_9deg',
        merge_test_val=False,
        loss_func=torch.nn.L1Loss()   
    )

    train_loader = tm.train_loader
    val_loader = tm.val_loader

    val_losses = []
    for epoch in trange(300):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):

                # get the inputs; data is a list of [inputs, labels]
                inputs = data
                inputs = inputs.to(device)

                labels = inputs.y

                # zero the parameter gradients
                tm.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = tm.loss_func(outputs, labels)
                loss.backward()
                tm.optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_steps += 1
                out = {
                    'train_loss': running_loss / (i + 1), 
                    'train_loss_degrees': running_loss / (i + 1) * 180/pi, 
                    'val_loss': val_losses[epoch - 1] if epoch > 0 else 'na',
                    'val_loss_degrees': val_losses[epoch - 1] * 180/pi if epoch > 0 else 'na',
                }
                
                tepoch.set_postfix(out)

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs = data
                inputs = inputs.to(device)

                labels = inputs.y

                # zero the parameter gradients

                # forward + backward + optimize
                outputs = net(inputs)
                loss = tm.loss_func(outputs, labels)
                tm.optimizer.step()

                # print statistics
                running_loss += loss.item()

                val_loss += loss.cpu().numpy()
                val_steps += 1

        val_loss /= val_steps
            
        val_losses.append(val_loss)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), tm.optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")



def main(num_samples=10, max_num_epochs=300, gpus_per_trial=2):
    config = {
        "pooling_size": tune.choice([(16/346, 12/260), (8/346, 3/260), (32/346, 24/260)]), 
        "pooling_outputs": tune.choice([32,64,128]), 
        "pooling_after_conv2": tune.choice([False, True]), 
        "more_layer": tune.choice([False, True]), 
        "more_block": tune.choice([False, True])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar, data_dir=None, checkpoint_dir='/home/hussain/tactile/ray_results'),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = model2(**best_trial.config["l1"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)



if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=1, max_num_epochs=1, gpus_per_trial=1)
