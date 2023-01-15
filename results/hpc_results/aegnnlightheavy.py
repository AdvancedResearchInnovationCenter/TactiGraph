from __future__ import division
import os.path as osp
import glob
import torch
import torch_geometric.transforms as T
import os
import json 
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch.utils.data
from torch_geometric.data import Data, DataLoader, Dataset

import numpy as np

import torch
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from collections.abc import Sequence

import os.path as osp
from pathlib import Path
import math

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph, knn_graph
from torch_geometric.utils import remove_isolated_nodes

#generate label
possible_angle = [math.radians(i) for i in range(1,11)]#[0.0174532925, 0.034906585, 0.0523598776, 0.075, 0.095, 0.115, 0.135, 0.15]#
N_examples = 20
list_of_rotations = [[0, 0, 0]]

for i in range(1, N_examples):
    theta = i * 2 * math.pi/(N_examples - 1)
    for phi in possible_angle:
        rx = phi * math.cos(theta)
        ry = phi * math.sin(theta)
        rotvec = [rx, ry, 0]
        list_of_rotations.append(rotvec)


cases_dict = {i+1: list_of_rotations[i][:2] for i in range(len(list_of_rotations))}
cases_dict[0] = [0, 0]

im_width=346
im_height=260

def files_exist(files):
    return all([osp.exists(f) for f in files])


class TactileDataset(Dataset):
    """_summary_

        Args:
            root (_type_): _description_
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
            features (str, optional): _description_. Defaults to 'all'.
            reset (bool, optional): _description_. Defaults to False.
    """

    def __init__(self, root, transform=None, pre_transform=None, features='all', reset=False, augment=False):
        """_summary_

        Args:
            root (_type_): _description_
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
            features (str, optional): _description_. Defaults to 'all'.
            reset (bool, optional): _description_. Defaults to False.
        """
        if reset:
            print('rm -rf ' + root + '/processed')
            ret=os.system('rm -rf ' + root + '/processed')
        root = Path(root)

        assert features in ['pol', 'coords', 'all', 'pol_time']
        self.features = features

        self.augment = augment
        super(TactileDataset, self).__init__(root, transform, pre_transform)
        self._indices = None
        
    def indices(self) -> Sequence:
        return range(self.len()) if self._indices is None else self._indices

    @property
    def raw_file_names(self):
        filenames = os.path.join(self.raw_dir, 'contact_cases.json')
        file = [f.split('/')[-1] for f in filenames]
        #print(file)
        return file

    @property
    def processed_file_names(self):
        #glob.glob(os.path.join(self.raw_dir,'../processed/', '*_.pt'))
        filenames = glob.glob(str(self.root / 'processed' / 'sample_*.pt'))
        file = [f.split('/')[-1] for f in filenames]
        saved_file = [f.replace('.pt','.pt') for f in file]
        return saved_file

    def __len__(self):
        return len(self.processed_file_names)

    def indices(self) -> Sequence:
        return range(self.__len__()) if self._indices is None else self._indices

    def sample_generator(self, samples_):
        for key, sample in samples_.items():
            case = sample['case']
            event_array = np.array(sample['events'])
            if not self.augment:
               yield case, event_array 
            else:
                for angle in [0, 90, 180, 270]:
                    if angle == 0:
                        yield case, event_array
                    else:
                        yield rotate_case(event_array, case, angle)

    def process(self):
        knn = 32
        with open(self.root / 'raw' / 'contact_cases.json', 'r') as f:
            samples_ = json.load(f)

        samples = samples_
        if self.augment:
            samples = {}
            for i, (case, event_array) in enumerate(self.sample_generator(samples_)):
                samples[f'sample_{i+1}'] = {
                    'events': event_array,
                    'case': case
                }
                

        for sample_id in samples.keys():
            events = np.array(samples[sample_id]['events'])

            coord1, coord2 = torch.tensor(events[:, 0:2].astype(np.float32)).T 
            ts = events[:, 2]
            ts = ((ts - ts.min()) / (ts.max() - ts.min())).astype(np.float32)
            coord3 = torch.tensor(ts)
            pos = torch.stack((coord1 / im_width, coord2 / im_height, coord3)).T

            if self.features == 'pol':
                feature = torch.tensor(events[:, 3].astype(np.float32))
                feature = feature.view(-1, 1)
            elif self.features == 'coords':
                feature = torch.stack((coord1 / im_width, coord2 / im_height, coord3)).T
            elif self.features == 'pol_time':
                feature = torch.stack((
                    torch.tensor(events[:, 3].astype(np.float32)),
                    coord3 
                )).T
            elif self.features == 'all':
                feature = torch.hstack((
                    torch.stack((coord1 / im_width, coord2 / im_height, coord3)).T, 
                    torch.tensor(events[:, 3].astype(np.float32)).reshape(-1, 1)
                    ))

            case = samples[sample_id]['case']

            #edge_index = radius_graph(pos, r=0.1, max_num_neighbors=10)
            edge_index = knn_graph(pos, knn)
            if self.features == 'pol_time':
                pos = pos[:, :2]

            #edge_index, _, mask = remove_isolated_nodes(edge_index=edge_index, num_nodes=feature.shape[0])

            #print(edge_index, sum(mask))
            #print(mask.shape, data.x.shape, data.edge_index.shape)
            
            pseudo_maker = T.Cartesian(cat=False, norm=True)
            

            y = torch.tensor(np.array(cases_dict[case], dtype=np.float32)).reshape(1, -1)

            data = Data(x=feature, edge_index=edge_index, pos=pos, y=y)
            data = pseudo_maker(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                    continue

            if self.pre_transform is not None:
                    data = self.pre_transform(data)

            torch.save(data, self.root / 'processed' / f'{sample_id}.pt')
        """
        with open(self.root.parent / 'extraction_params.json', 'r') as f:
            params = json.load(f)
        
        params['kNN'] = knn
        params['node_features'] = self.features
        print(params)
        with open(self.root.parent / 'extraction_params.json', 'w') as f:
            json.dump(params, f, indent=4)
        """
            
            

    def get(self, idx):
        # print("I'm in get ", self.processed_dir)

        data = torch.load(osp.join(self.processed_paths[idx]))
        return data

    def load_all_raw(self):
        samples = {}
        for subset in ['train', 'val', 'test']:
            with open(self.root.parent / subset / 'raw' / 'contact_cases.json', 'r') as f:
                subset_samples = json.load(f)
                subset_samples_tot_idx = {item[1]['total_idx']: 
                                        {
                                            'events': item[1]['events'], 
                                            'case': item[1]['case']
                                        } for item in subset_samples.items()}
            
            samples.update(subset_samples_tot_idx)
        return samples


import json
import torch
import torch_geometric as pyg
from tqdm.auto import tqdm, trange  
from pathlib import Path
from numpy import pi
from pandas import DataFrame

class TrainModel():

    def __init__(
        self, 
        extraction_case_dir, 
        model,
        n_epochs = 150,
        optimizer = 'adam',
        lr = 0.001,
        loss_func = torch.nn.L1Loss(),
        transform = None,
        features = 'all',
        weight_decay=0,
        patience=10,
        batch = 1,
        augment=False
        ):

        self.extraction_case_dir = Path(extraction_case_dir)
        self.transform = transform

        self.train_data = TactileDataset(self.extraction_case_dir / 'train', transform=transform, features=features, augment=augment)
        self.val_data = TactileDataset(self.extraction_case_dir / 'val', features=features)
        self.test_data = TactileDataset(self.extraction_case_dir / 'test', features=features)

        self.train_loader = pyg.loader.DataLoader(self.train_data, shuffle=True, batch_size=batch)
        self.val_loader = pyg.loader.DataLoader(self.val_data)
        self.test_loader = pyg.loader.DataLoader(self.test_data)

        self.model = model
        self.n_epochs = n_epochs


        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError('use tm.optimizer = torch.optim.<optimizer>')
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr=1e-5, patience=patience)

        self.loss_func = loss_func

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        

    def train(self):
        self.train_losses = []
        self.val_losses = []
        self.lr = []

        name = str(type(self.model)).split('.')[-1][:-2]
        path = Path('results') / name

        for epoch in trange(self.n_epochs, desc='training', unit='epoch'):
            #bunny(epoch)
            epoch_loss = 0
            
            if (epoch == 10):
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.001
    
            if epoch == 110:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.0001

            if epoch == 200:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.00001
                    
            lr = self.optimizer.param_groups[0]['lr']
            self.lr.append(lr)
            val_loss = torch.inf
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for i, data in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    with torch.autograd.detect_anomaly():
                        data = data.to(self.device)
                        self.optimizer.zero_grad()
                        end_point = self.model(data)
                        loss = self.loss_func(end_point, data.y)
                        loss.backward()
                        self.optimizer.step()
                        lr = self.optimizer.param_groups[0]['lr']

                        epoch_loss += loss.detach().item()
                    
                        tepoch.set_postfix({
                            'train_loss': epoch_loss / (i + 1), 
                            'train_loss_degrees': epoch_loss / (i + 1) * 180/pi, 
                            'val_loss': self.val_losses[epoch - 1] if epoch > 0 else 'na',
                            'val_loss_degrees': self.val_losses[epoch - 1] * 180/pi if epoch > 0 else 'na',
                            'lr': lr
                            })

                #self.scheduler.step(val_loss)
                epoch_loss /= len(self.train_data)
                val_loss = self.validate()
                tepoch.set_postfix({'train_loss': epoch_loss, 'val_loss': val_loss})
                self.train_losses.append(epoch_loss)
                self.val_losses.append(val_loss)
            if (epoch + 1) % 1 == 0:
                self.log(current_epoch=epoch)


        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.model, path / 'model_ckp.pt')
        torch.save(self.model, path / 'model.pt')

    def validate(self):
        loss = 0
        for i, data in enumerate(self.val_loader):      
            data = data.to(self.device)
            end_point = self.model(data)

            loss += self.loss_func(end_point, data.y).detach().item()
        loss /= len(self.val_data)
        return loss
    
    def test(self):
        loss = 0
        for i, data in enumerate(self.test_loader):      
            data = data.to(self.device)
            end_point = self.model(data)

            loss += self.loss_func(end_point, data.y).detach().item()
        loss /= len(self.train_data)
        return loss

    def augment(self, batch):
        pass

    def log(self, current_epoch):
        #find model name
        print('logging')
        name = str(type(self.model)).split('.')[-1][:-2]
        path = Path('results') / name
        if not path.exists():
            path.mkdir(parents=True)

        with open(path / 'training_params.json', 'w') as f:
            params = {
                'model': name,
                'extraction_used': str(self.extraction_case_dir),
                'n_epochs': self.n_epochs,
                'final_val_loss_degrees': self.val_losses[-1] * 180 / pi,
            }
            json.dump(params, f, indent=4)

        train_log = { 
            'epoch': [i for i in range(1, current_epoch+2)],
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'lr': self.lr
        }
        DataFrame(train_log).to_csv(path / 'train_log.csv', index=False)

import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool_x, voxel_grid
from typing import List, Optional, Tuple, Union


class MaxPoolingX(torch.nn.Module):

    def __init__(self, voxel_size: List[int], size: int):
        super(MaxPoolingX, self).__init__()
        self.voxel_size = voxel_size
        self.size = size

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:
        cluster = voxel_grid(pos, batch=batch, size=self.voxel_size)
        x, _ = max_pool_x(cluster, x, batch, size=self.size)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size}, size={self.size})"

import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool, voxel_grid
from typing import Callable, List, Optional, Tuple, Union


class MaxPooling(torch.nn.Module):

    def __init__(self, size: List[int], transform: Callable[[Data, ], Data] = None):
        super(MaxPooling, self).__init__()
        self.voxel_size = list(size)
        self.transform = transform

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None, return_data_obj: bool = False
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:
        assert edge_index is not None, "edge_index must not be None"

        cluster = voxel_grid(pos[:, :2], batch=batch, size=self.voxel_size)
        data = Data(x=x, pos=pos, edge_index=edge_index, batch=batch)
        data = max_pool(cluster, data=data, transform=self.transform)  # transform for new edge attributes
        if return_data_obj:
            return data
        else:
            return data.x, data.pos, getattr(data, "batch"), data.edge_index, data.edge_attr

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size})"

import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import elu
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian


class GraphRes_lightheavy(torch.nn.Module):

    def __init__(self):
        super(GraphRes_lightheavy, self).__init__()
        dim = 3

        bias = False
        root_weight = False
        pooling_size=(16/346, 12/260)

        # Set dataset specific hyper-parameters.
        kernel_size = 2
        n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
        pooling_outputs = 32
        #kernel_size = 8
        #n = [1, 16, 32, 32, 32, 128, 128, 128]
        #pooling_outputs = 128

        self.conv1 = SplineConv(n[0], n[1], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm1 = BatchNorm(in_channels=n[1])
        self.conv2 = SplineConv(n[1], n[2], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm2 = BatchNorm(in_channels=n[2])

        self.conv3 = SplineConv(n[2], n[3], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm3 = BatchNorm(in_channels=n[3])
        self.conv4 = SplineConv(n[3], n[4], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm4 = BatchNorm(in_channels=n[4])

        self.conv5 = SplineConv(n[4], n[5], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm5 = BatchNorm(in_channels=n[5])
        self.pool5 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

        self.conv6 = SplineConv(n[5], n[6], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm6 = BatchNorm(in_channels=n[6])
        self.conv7 = SplineConv(n[6], n[7], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm7 = BatchNorm(in_channels=n[7])

        self.pool7 = MaxPoolingX(0.25, size=16)
        self.fc = Linear(pooling_outputs * 16, out_features=2, bias=bias)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm1(data.x)
        data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2(data.x)

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

        x = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)

model = GraphRes_lightheavy()
print(model)
tm = TrainModel('data/extractions/contact_extraction2/', model.cuda(), batch=4, lr=0.01, n_epochs=300, features='pol')
tm.train()