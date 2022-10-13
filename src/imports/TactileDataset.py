
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

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph, knn_graph
from imports.extract_contact_cases import cases_dict

im_height=260
im_width=346

def files_exist(files):
    return all([osp.exists(f) for f in files])


class TactileDataset(Dataset):

    def __init__(self, root, transform=None, pre_transform=None):
        """Generates TactileDataset for loading

        Args:
            root (string): directory of data
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
        """
        root = Path(root)
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

    def process(self):
        with open(self.root / 'raw' / 'contact_cases.json', 'r') as f:
            samples = json.load(f)

        for sample_id in samples.keys():
            events = np.array(samples[sample_id]['events'])
            feature = torch.tensor(events[:, 3])
            case = samples[sample_id]['case']

            coord1, coord2 = torch.tensor(events[:, 0:2]).double().T 

            ts = events[:, 2]
            coord3 = torch.tensor((ts - ts.min()) / (ts.max() - ts.min()))
            pos = torch.stack((coord1 / im_width, coord2 / im_height, coord3)).T
            edge_index = knn_graph(pos, 10)
            y = torch.tensor(cases_dict[case])

            data = Data(x=feature, edge_index=edge_index, pos=pos, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                    continue

            if self.pre_transform is not None:
                    data = self.pre_transform(data)

            torch.save(data, self.root / 'processed' / f'{sample_id}.pt')

    def get(self, idx):
        # print("I'm in get ", self.processed_dir)

        data = torch.load(osp.join(self.processed_paths[idx]))
        return data

    def load_all_raw(self):
        samples = {}
        for subset in ['train', 'val', 'test']:
            with open(self.root / 'raw' / 'contact_cases.json', 'r') as f:
                import json
                subset_samples = json.load(f)
                subset_samples_tot_idx = {item[1]['total_idx']: 
                                        {
                                            'events': item[1]['events'], 
                                            'case': item[1]['case']
                                        } for item in subset_samples.items()}
            
            samples.update(subset_samples_tot_idx)
        return samples