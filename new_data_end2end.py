#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import models
import numpy as np
import sys, os
import rospy
import rosbag
import math
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
import time 
from tqdm.auto import tqdm

#parameters of processing
#start after 1672235485038031343
frequency = 30 #Hz
bag_file_name = '/media/hussain/drive1/2022-12-28-17-51-22.bag'#'/home/hussain/me/projects/tactile/data/dataset_ENVTACT_new2.bag'
time_window_size = 6
examples_per_edge = 1

bag_file = rosbag.Bag(bag_file_name)
# In[2]:


events = []
contact_status = []
contact_status_ts = []
contact_case = [] #0:No contact 1: center, 2:remainder of contacts as in list_of_rotations
contact_case_ts = []
contact_angle = []

contact_case_updated = []
contact_case_updated_ts = []

event_packet_size = []
event_packet_size_ts = []
contact_angle = []

#[math.radians(i) for i in range(1,11)]#

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

print(len(list_of_rotations))
print(list_of_rotations)

from rospy import Time
start_time = 1672235485038031343*1e-9
#parse from 1672235485038031343 onwards
events = []
contact_status = []
contact_status_ts = []
contact_angle = []
contact_angle = []

topics = ['/contact_status', '/contact_angle']

for topic, msg, t in tqdm(
    bag_file.read_messages(topics=topics, start_time=Time(start_time)), 
    total=sum([bag_file.get_message_count(top) for top in topics]),
    desc='parsing bag',
    unit='msg'
):
    if topic == '/dvs/events':
        for e in msg.events:
            event = [e.x, e.y, e.ts.to_nsec(), e.polarity]
            events.append(event)
    elif topic == '/contact_status':
        contact_status.append(msg.data)
        contact_status_ts.append(t.to_nsec())
    elif topic == '/contact_angle':
        contact_angle.append([msg.x, msg.y, msg.z])

        # Updated contact status according to no. of events

#print(events)
bag_file.close()


# In[3]:


import pandas as pd

df = pd.read_csv('parsed_bag1.csv')


# In[4]:


import matplotlib.pyplot as plt
plt.figure(figsize=(150, 40))
plt.plot(df['ts'], df['contact_status'])


# In[5]:


def find_case(idx):
    if not contact_status[idx]:
        return 0
    else:
        best_rot_diff = 100
        best_rot_idx = 1
        i = 1
        x, y, z = df[['contact_angle_x', 'contact_angle_y', 'contact_angle_z']].iloc[idx]
        for rot in list_of_rotations:
            diff_vals = np.sqrt(np.power(rot[0] - x, 2) +  np.power(rot[1] - y, 2) + np.power(rot[2] - z, 2))
            if best_rot_diff > diff_vals:
                best_rot_diff = diff_vals
                best_rot_idx = i
            i = i + 1
        return best_rot_idx


# In[6]:


plt.figure(figsize=(350, 40))
plt.plot(df['ts'], df['contact_case'])


# In[7]:


import h5py
import dask.array as da

h5_arr = h5py.File('events2.h5', 'r')
events = da.array(h5_arr['events'])
events


# In[8]:


case_span = 2.66e9
find_ts_idx = lambda ts: np.searchsorted(df['ts'], ts)

def look_ahead_big(ts, idx_ts):
    fin_ts = ts + case_span
    fin_idx = find_ts_idx(fin_ts)
    print(fin_idx)
    if df['contact_status'][fin_idx]:
        #look further
        more = True
        fin_idx_ = fin_idx 
        while more:
            fin_idx_ += 1
            if fin_idx_ - fin_idx > 25:
                print('warning more than 25 idx away from init_ts + case_span')
            if df['contact_status'][fin_idx_]:
                continue
            else:
                more = False
        print(f'was before case ended by {fin_idx_ - fin_idx} indexes')
        fin_idx = fin_idx_ - 1
    else:
        #look backwards
        more = True
        fin_idx_ = fin_idx 
        while more:
            fin_idx_ -= 1
            if not df['contact_status'][fin_idx_]:
                continue
            else:
                more = False
        print(f'was ahead case ended by {fin_idx - fin_idx} indexes')
        fin_idx = fin_idx_ + 1
        
    return df['ts'][fin_idx]


def find_case(ts):
    idx = find_ts_idx(ts)
    best_rot_diff = 100
    best_rot_idx = 1
    i = 1
    x, y, z = df[['contact_angle_x', 'contact_angle_y', 'contact_angle_z']].iloc[idx]
    for rot in list_of_rotations:
        diff_vals = np.sqrt(np.power(rot[0] - x, 2) +  np.power(rot[1] - y, 2) + np.power(rot[2] - z, 2))
        if best_rot_diff > diff_vals:
            best_rot_diff = diff_vals
            best_rot_idx = i
        i = i + 1
    return best_rot_idx


i = 0 
cases_ts = []
cases_idx = []
cases = []
pbar = tqdm(total=len(df['contact_status']), desc='extracting contact timestamps')

while i < len(df['contact_status']):
    if df['contact_status'][i]:
        init_ts = df['ts'][i]
        fin_ts = look_ahead_big(init_ts, i)
        fin_idx = find_ts_idx(fin_ts)
        case = find_case(np.mean([init_ts, fin_ts]))
        
        cases.append(case)
        cases_ts.append([init_ts, fin_ts])
        cases_idx.append([i, fin_idx])
        print(len(cases_ts), init_ts, fin_ts, (fin_ts - init_ts)*1e-9, i, fin_idx, case, '\n')
        i = fin_idx + 1
        pbar.update(fin_idx + 1 - i)

    else:
        i += 1
        pbar.update(1)


# In[9]:


contact_case_alt = np.zeros(len(df))
for i, case in enumerate(cases):
    contact_case_alt[cases_idx[i][0]:cases_idx[i][1] + 1] = case
    
plt.figure(figsize=(200, 40))
plt.plot(df['ts'], contact_case_alt)


# In[10]:


plt.hist([i[1]-i[0] for i in cases_ts])


# In[11]:


plt.hist([i[1]-i[0] for i in cases_idx])


# In[12]:


def ev_ts2idx(ts, guess_idx, init_stepsize=50000):
    stepsize = init_stepsize
    search_sorted = lambda st: da.searchsorted(events[guess_idx:guess_idx+st, 2], da.array([ts])).compute()[0]
    n_iters = 0
    
    out = search_sorted(stepsize)
    
    while out == stepsize:
        n_iters += 1
        stepsize = 10*stepsize
        
        out = search_sorted(stepsize)
    if n_iters > 1:
        print(ts, n_iters)
    return out + guess_idx


# In[13]:


delta_t = 0.05e9
margin = 0#-0.025e9
dist_from_center = lambda x, y: np.sqrt((x - 180)**2 + (y - 117)**2)
circle_rad=85

event_arrays = []
ev_idx_init = []
label_contact_case = []

guess_idx = 0

for i, idx in enumerate(tqdm(cases_idx)):
    init_ts_idx = ev_ts2idx(cases_ts[i][0] + margin, guess_idx)
    ev_idx_init.append(init_ts_idx)
    guess_idx = init_ts_idx
    fin_ts_idx = ev_ts2idx(cases_ts[i][0] + delta_t, guess_idx)#da.searchsorted(events[:, 2], da.array([cases_ts[i][0] + delta_t]))
    guess_idx = fin_ts_idx
    if fin_ts_idx - init_ts_idx + 1 < 750:
        continue    
    else:
        event_array = events[init_ts_idx:fin_ts_idx+1].compute()

    in_circle = dist_from_center(event_array[:, 0], event_array[:, 1]) < circle_rad  
    event_arrays.append(event_array[in_circle, :])
    label_contact_case.append(cases[i])


# In[15]:


plt.hist(label_contact_case, bins=191)


# In[159]:


import seaborn as sns

idx = 100
plt.figure
sns.scatterplot(x=event_arrays[idx][:, 0], y=250-event_arrays[idx][:, 1], hue=event_arrays[idx][:, 3])
event_arrays[idx].shape


# In[16]:


plt.figure(figsize=(40, 5))
plt.hist([i.shape[0] for i in event_arrays], bins=30)
plt.xticks(np.arange(12000, step=500));

(array([187519441,        46,        31,        20,        14,        10,
                4,         4,         4,         2,         2,         1,
                0,         3,         1,         0,         0,         1,
                0,         1,         0,         0,         0,         0,
              120,       161,         0,         0,         0,         0,
                0,         0,         0,         0,         0,         0,
                0,         0,         0,         0,         0,         0,
                0,         0,         0,         0,         0,         0,
                0,         1]),
 array([0.        , 0.00532326, 0.01064651, 0.01596977, 0.02129303,
        0.02661629, 0.03193954, 0.0372628 , 0.04258606, 0.04790932,
        0.05323257, 0.05855583, 0.06387909, 0.06920235, 0.0745256 ,
        0.07984886, 0.08517212, 0.09049538, 0.09581863, 0.10114189,
        0.10646515, 0.11178841, 0.11711166, 0.12243492, 0.12775818,
        0.13308144, 0.13840469, 0.14372795, 0.14905121, 0.15437447,
        0.15969772, 0.16502098, 0.17034424, 0.1756675 , 0.18099075,
        0.18631401, 0.19163727, 0.19696053, 0.20228378, 0.20760704,
        0.2129303 , 0.21825356, 0.22357681, 0.22890007, 0.23422333,
        0.23954659, 0.24486984, 0.2501931 , 0.25551636, 0.26083961,
        0.26616287]))
# In[17]:


samples={}

gen = zip(label_contact_case, event_arrays) 
for i, (case, event_array) in enumerate(gen):
    samples[f'sample_{i+1}'] = {
        'events': event_array.tolist(),
        'case': case
        }


# In[18]:


from pathlib import Path
import json

EXTRACTIONS_DIR = Path('./data/extractions/').resolve()
outdir = EXTRACTIONS_DIR / 'contact_extraction2'

if not outdir.exists():
    outdir.mkdir(parents=True)
    

with open(outdir / 'samples.json', 'w') as f:
    json.dump(samples, f, indent=4)


# In[36]:


from sklearn.model_selection import train_test_split


sample_idx = list(samples.keys())
cases = [str(samples[s_idx]['case']) for s_idx in sample_idx]

train_idx, val_test_idx = train_test_split(sample_idx, test_size=1-0.7, random_state=0) #fixed across extractions

cases = [str(samples[s_idx]['case']) for s_idx in val_test_idx]
val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, random_state=0) #fixed across extractions

print(len(train_idx), len(val_idx), len(test_idx))
subsets = zip(['train', 'test', 'val'], [train_idx, val_idx, test_idx])

if not outdir.exists():
    outdir.mkdir(parents=True)

for sub_name, subset in subsets:
    if not (outdir / sub_name).exists():
        (outdir / sub_name / 'raw').mkdir(parents=True)
        (outdir / sub_name / 'processed').mkdir(parents=True)
    with open(outdir / sub_name / 'raw' / 'contact_cases.json', 'w') as f:
        subset_samples = {}
        for i, subset_idx in enumerate(subset):
            sample = samples[subset_idx]
            sample['total_idx'] = subset_idx
            subset_samples[f'sample_{i+1}'] = sample
        json.dump(subset_samples, f, indent=4)


# In[23]:


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
from torch_geometric.utils import remove_isolated_nodes


# In[24]:


cases_dict = {i+1: list_of_rotations[i][:2] for i in range(len(list_of_rotations))}
cases_dict[0] = [0, 0]
cases_dict


# In[40]:


def files_exist(files):
    return all([osp.exists(f) for f in files])

im_height=260
im_width=346

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


# In[43]:


train_td = TactileDataset('data/extractions/contact_extraction2/train')
val_td = TactileDataset('data/extractions/contact_extraction2/val')


# In[45]:


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

                self.scheduler.step(val_loss)
                epoch_loss /= len(self.train_data)
                val_loss = self.validate()
                tepoch.set_postfix({'train_loss': epoch_loss, 'val_loss': val_loss})
                self.train_losses.append(epoch_loss)
                self.val_losses.append(val_loss)
            if (epoch + 1) % 1 == 0:
                self.log(current_epoch=epoch)
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



# In[48]:


import torch
from torch_geometric.nn import SplineConv, voxel_grid, max_pool, max_pool_x
import torch.nn.functional as F
import torch_geometric.transforms as T

class spline(torch.nn.Module):
    def __init__(self):
        super(spline, self).__init__()
        self.conv1 = SplineConv(4, 64, dim=3, kernel_size=3)
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


# In[50]:


model = spline()
model


# In[56]:


tm = TrainModel('data/extractions/contact_extraction2/', model.cuda(), batch=4)


# In[ ]:


tm.train()

