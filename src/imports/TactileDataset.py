
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
from imports.ExtractContactCases import cases_dict
from torch_geometric.utils import remove_isolated_nodes

im_height=260
im_width=346

possible_angle = [0.0174532925, 0.034906585, 0.0523598776, 0.075, 0.095, 0.115, 0.135, 0.15]#
N_examples = 17
list_of_rotations = [[0, 0, 0]]

for i in range(1, N_examples):
    theta = i * 2 * np.pi/(N_examples - 1)
    for phi in possible_angle:
        rx = phi * np.cos(theta)
        ry = phi * np.sin(theta)
        rotvec = [rx, ry, 0]
        list_of_rotations.append(rotvec)

cases_dict = {i+1: list_of_rotations[i][:2] for i in range(len(list_of_rotations))}
cases_dict[0] = [0, 0]

def rotate_case(ev_arr, label, angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    centered = ev_arr[:, :2] - np.array([157, 124])
    rot_ev = (R @ centered.T).T + np.array([157, 124])
    
    rot_v = np.array(cases_dict[label])
    new_rot_v = R @ rot_v
    #print(new_rot_v, cases_dict[label])

    best_rot_diff = 100
    best_rot_idx = 1
    i = 1
    
    for rot in list_of_rotations:
        diff_vals = np.sqrt( np.power(rot[0] - new_rot_v[0], 2) +  np.power(rot[1] - new_rot_v[1], 2))
        if best_rot_diff > diff_vals:
            best_rot_diff = diff_vals
            best_rot_idx = i
        i = i + 1
    
    return best_rot_idx, np.concatenate([rot_ev.astype(int), ev_arr[:, 2:]], -1)

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

        assert features in ['pol', 'coords', 'all']
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
            elif self.features == 'all':
                feature = torch.hstack((
                    torch.stack((coord1 / im_width, coord2 / im_height, coord3)).T, 
                    torch.tensor(events[:, 3].astype(np.float32)).reshape(-1, 1)
                    ))

            case = samples[sample_id]['case']

            #edge_index = radius_graph(pos, r=0.1, max_num_neighbors=10)
            edge_index = knn_graph(pos, knn)

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

        with open(self.root.parent / 'extraction_params.json', 'r') as f:
            params = json.load(f)
        
        params['kNN'] = knn
        params['node_features'] = self.features
        print(params)
        with open(self.root.parent / 'extraction_params.json', 'w') as f:
            json.dump(params, f, indent=4)
            
            

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